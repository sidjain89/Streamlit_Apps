import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures

st.title("Feature Engineering App - Full Advanced Pipeline Builder")

# --- Upload dataset ---
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --- Specify column types ---
    st.subheader("Specify Column Types")
    all_columns = df.columns.tolist()
    continuous_cols = st.multiselect("Continuous (numeric) columns", all_columns)
    categorical_cols = st.multiselect("Categorical columns", all_columns)
    datetime_cols = st.multiselect("Datetime columns", all_columns)

    # Convert datetime columns
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # --- Specify target column for target encoding ---
    st.subheader("Target Column (for target encoding)")
    target_col = st.selectbox("Select target column (numeric)", [None]+all_columns)

    # --- Session state ---
    if "original_df" not in st.session_state:
        st.session_state.original_df = df.copy()  # preserve original
    if "engineered_df" not in st.session_state:
        st.session_state.engineered_df = df.copy()
    if "queue" not in st.session_state:
        st.session_state.queue = []

    df_transformed = st.session_state.engineered_df.copy()

    # --- Add column to queue ---
    st.subheader("Add Transformation to Queue")
    selectable_cols = continuous_cols + categorical_cols + datetime_cols
    col_to_engineer = st.selectbox("Select column", selectable_cols)

    # Missing value handling and transformation
    if col_to_engineer in continuous_cols:
        missing_strategy = st.selectbox("Missing value strategy", ["None", "Mean", "Median", "Fill with 0"])
        transformation = st.selectbox("Numeric transformation", 
                                      ["None", "StandardScaler", "MinMaxScaler", "Log", "Sqrt", "Polynomial Features"])
        degree = 2
        if transformation == "Polynomial Features":
            degree = st.number_input("Polynomial Degree", min_value=2, max_value=5, value=2)
    elif col_to_engineer in categorical_cols:
        missing_strategy = st.selectbox("Missing value strategy", ["None", "Mode", "Fill with 'Unknown'"])
        transformation = st.selectbox("Categorical transformation", ["None", "OneHot", "Label Encoding", "Target Encoding"])
    elif col_to_engineer in datetime_cols:
        missing_strategy = st.selectbox("Missing value strategy", ["None", "Forward Fill", "Backward Fill"])
        transformation = st.multiselect("Datetime extraction", ["Year", "Month", "Day", "Weekday", "Quarter"])

    # Add to queue
    if st.button("Add to Queue"):
        st.session_state.queue.append({
            "Column": col_to_engineer,
            "Missing": missing_strategy,
            "Transformation": transformation,
            "Degree": degree if 'degree' in locals() else None
        })
        st.success(f"Added {col_to_engineer} to queue!")

    # --- Display queue with delete option ---
    st.subheader("Transformation Queue")
    if st.session_state.queue:
        for idx, item in enumerate(st.session_state.queue):
            col1, col2 = st.columns([4,1])
            col1.write(f"{idx+1}. Column: {item['Column']}, Transformation: {item['Transformation']}, Missing: {item['Missing']}")
            if col2.button("Delete", key=f"del_{idx}"):
                st.session_state.queue.pop(idx)
                st.session_state.engineered_df = st.session_state.original_df.copy()  # reset engineered_df
                st.success("Deleted item from queue. Changes will take effect on next apply.")
                break  # break to prevent index issues while looping

    # --- Apply queued transformations ---
    if st.button("Apply All Queued Transformations"):
        df_transformed = st.session_state.original_df.copy()  # start from original

        for step in st.session_state.queue:
            col = step["Column"]
            miss = step["Missing"]
            trans = step["Transformation"]
            deg = step.get("Degree", 2)

            # Handle missing values
            if miss != "None":
                if col in continuous_cols:
                    if miss == "Mean":
                        df_transformed[col+"_filled"] = df_transformed[col].fillna(df_transformed[col].mean())
                    elif miss == "Median":
                        df_transformed[col+"_filled"] = df_transformed[col].fillna(df_transformed[col].median())
                    elif miss == "Fill with 0":
                        df_transformed[col+"_filled"] = df_transformed[col].fillna(0)
                elif col in categorical_cols:
                    if miss == "Mode":
                        df_transformed[col+"_filled"] = df_transformed[col].fillna(df_transformed[col].mode()[0])
                    elif miss == "Fill with 'Unknown'":
                        df_transformed[col+"_filled"] = df_transformed[col].fillna("Unknown")
                elif col in datetime_cols:
                    if miss == "Forward Fill":
                        df_transformed[col+"_filled"] = df_transformed[col].fillna(method='ffill')
                    elif miss == "Backward Fill":
                        df_transformed[col+"_filled"] = df_transformed[col].fillna(method='bfill')
            else:
                df_transformed[col+"_filled"] = df_transformed[col]

            # Apply transformations
            if col in continuous_cols:
                base_col = col+"_filled"
                if trans == "StandardScaler":
                    df_transformed[col+"_std"] = StandardScaler().fit_transform(df_transformed[[base_col]])
                elif trans == "MinMaxScaler":
                    df_transformed[col+"_minmax"] = MinMaxScaler().fit_transform(df_transformed[[base_col]])
                elif trans == "Log":
                    df_transformed[col+"_log"] = np.log1p(df_transformed[base_col])
                elif trans == "Sqrt":
                    df_transformed[col+"_sqrt"] = np.sqrt(df_transformed[base_col])
                elif trans == "Polynomial Features":
                    poly = PolynomialFeatures(degree=deg, include_bias=False)
                    poly_features = poly.fit_transform(df_transformed[[base_col]])
                    poly_cols = [f"{col}_poly_{i+1}" for i in range(poly_features.shape[1])]
                    df_poly = pd.DataFrame(poly_features, columns=poly_cols)
                    df_transformed = pd.concat([df_transformed.reset_index(drop=True), df_poly], axis=1)

            elif col in categorical_cols:
                base_col = col+"_filled"
                if trans == "Label Encoding":
                    df_transformed[col+"_label"] = LabelEncoder().fit_transform(df_transformed[base_col])
                elif trans == "OneHot":
                    df_transformed = pd.get_dummies(df_transformed, columns=[base_col])
                elif trans == "Target Encoding" and target_col:
                    df_transformed[col+"_TE"] = df_transformed.groupby(base_col)[target_col].transform('mean')

            elif col in datetime_cols:
                base_col = col+"_filled"
                if isinstance(trans, str):
                    trans = [trans]
                for ext in trans:
                    if ext == "Year":
                        df_transformed[col+"_Year"] = df_transformed[base_col].dt.year
                    elif ext == "Month":
                        df_transformed[col+"_Month"] = df_transformed[base_col].dt.month
                    elif ext == "Day":
                        df_transformed[col+"_Day"] = df_transformed[base_col].dt.day
                    elif ext == "Weekday":
                        df_transformed[col+"_Weekday"] = df_transformed[base_col].dt.weekday
                    elif ext == "Quarter":
                        df_transformed[col+"_Quarter"] = df_transformed[base_col].dt.quarter

        st.session_state.engineered_df = df_transformed
        st.session_state.queue = []
        st.success("All queued transformations applied!")

    # --- Preview transformed dataset BEFORE interactions ---
    st.subheader("Dataset Preview After Queued Transformations (Before Interactions)")
    st.dataframe(st.session_state.engineered_df.head())

    # --- Feature Interactions ---
    st.subheader("Create Feature Interactions (Optional)")

    # Preserve selections in session_state
    if "numeric_interactions" not in st.session_state:
        st.session_state.numeric_interactions = []
    if "cat_interactions" not in st.session_state:
        st.session_state.cat_interactions = []
    if "create_cat_combinations" not in st.session_state:
        st.session_state.create_cat_combinations = False

    st.session_state.numeric_interactions = st.multiselect(
        "Numeric × Numeric interactions",
        continuous_cols,
        default=st.session_state.numeric_interactions
    )
    st.session_state.cat_interactions = st.multiselect(
        "Categorical × Numeric interactions",
        categorical_cols,
        default=st.session_state.cat_interactions
    )
    st.session_state.create_cat_combinations = st.checkbox(
        "Categorical × Categorical interactions",
        value=st.session_state.create_cat_combinations
    )

    # Numeric operation selection
    numeric_operation = st.selectbox(
        "Select operation for numeric × numeric interactions",
        ["Multiply", "Add", "Subtract", "Divide"]
    )

    if st.button("Generate Feature Interactions"):
        df_inter = st.session_state.engineered_df.copy()

        # Numeric × Numeric with chosen operation
        for i in range(len(st.session_state.numeric_interactions)):
            for j in range(i+1, len(st.session_state.numeric_interactions)):
                col1 = st.session_state.numeric_interactions[i]
                col2 = st.session_state.numeric_interactions[j]
                new_col_name = f"{col1}_{numeric_operation}_{col2}"

                if numeric_operation == "Multiply":
                    df_inter[new_col_name] = df_inter[col1] * df_inter[col2]
                elif numeric_operation == "Add":
                    df_inter[new_col_name] = df_inter[col1] + df_inter[col2]
                elif numeric_operation == "Subtract":
                    df_inter[new_col_name] = df_inter[col1] - df_inter[col2]
                elif numeric_operation == "Divide":
                    df_inter[new_col_name] = df_inter[col1] / df_inter[col2].replace(0, np.nan)

        # Categorical × Numeric
        for cat_col in st.session_state.cat_interactions:
            if cat_col in df_inter.columns:
                cat_dummies = pd.get_dummies(df_inter[cat_col], prefix=cat_col)
                for num_col in continuous_cols:
                    for dummy_col in cat_dummies.columns:
                        df_inter[f"{dummy_col}_x_{num_col}"] = cat_dummies[dummy_col] * df_inter[num_col]

        # Categorical × Categorical
        if st.session_state.create_cat_combinations:
            for i in range(len(st.session_state.cat_interactions)):
                for j in range(i+1, len(st.session_state.cat_interactions)):
                    col1 = st.session_state.cat_interactions[i]
                    col2 = st.session_state.cat_interactions[j]
                    df_inter[f"{col1}_x_{col2}"] = df_inter[col1].astype(str) + "_" + df_inter[col2].astype(str)

        st.session_state.engineered_df = df_inter
        st.success("Feature interactions created!")

    # --- Final Preview & Download ---
    st.subheader("Final Engineered Dataset Preview")
    st.dataframe(st.session_state.engineered_df.head())
    st.download_button("Download Engineered CSV", st.session_state.engineered_df.to_csv(index=False), "engineered_data.csv")
