import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# Optional: SMOTE and undersampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

st.title("🌳 Decision Tree App")

# 1. Load Data
st.sidebar.header("1. Load Data")
data_source = st.sidebar.radio("Choose data source:", ["Upload CSV", "Use Iris Dataset"])

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame

if 'df' in locals():
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()

    # 2. Feature/Target Configuration
    st.sidebar.header("2. Configure Variables")
    target_column = st.sidebar.selectbox("Select target variable", all_columns)
    feature_columns = st.sidebar.multiselect("Select feature columns", [col for col in all_columns if col != target_column], default=[col for col in all_columns if col != target_column])

    if feature_columns:
        target_type = st.sidebar.selectbox("Target Variable Type", ["Categorical", "Continuous"])

        st.sidebar.markdown("### Feature Types")
        categorical_features = st.sidebar.multiselect("Categorical features", feature_columns, default=[])
        continuous_features = [col for col in feature_columns if col not in categorical_features]

        encoding_strategy = st.sidebar.radio("Encoding strategy", ["One-Hot Encoding", "Label Encoding"])

        # Resampling options
        st.sidebar.header("3. Class Imbalance Handling")
        resampling_method = st.sidebar.selectbox("Resampling method", ["None", "SMOTE (oversample)", "Random undersample"])

        df_clean = df[feature_columns + [target_column]].copy()
        df_encoded = df_clean.copy()

        # Encode categorical features
        if encoding_strategy == "One-Hot Encoding":
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_features)
        else:
            le_dict = {}
            for col in categorical_features:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                le_dict[col] = le

        # Encode target if classification
        if target_type == "Categorical":
            target_encoder = LabelEncoder()
            df_encoded[target_column] = target_encoder.fit_transform(df_encoded[target_column])

        X = df_encoded.drop(columns=[target_column])
        y = df_encoded[target_column]

        # Resampling (only for classification)
        if target_type == "Categorical" and resampling_method != "None":
            if resampling_method == "SMOTE (oversample)":
                sm = SMOTE(random_state=42)
                X, y = sm.fit_resample(X, y)
            elif resampling_method == "Random undersample":
                rus = RandomUnderSampler(random_state=42)
                X, y = rus.fit_resample(X, y)

        # 4. Train/Test Split
        st.sidebar.header("4. Train/Test Split")
        test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # 5. Model Parameters
        st.sidebar.header("5. Model Parameters")
        criterion_cls = st.sidebar.selectbox("Criterion (classification)", ["gini", "entropy"])
        criterion_reg = st.sidebar.selectbox("Criterion (regression)", ["squared_error", "friedman_mse", "absolute_error"])
        max_depth = st.sidebar.slider("Max depth", 1, 30, 5)
        min_samples_split = st.sidebar.slider("Min samples split", 2, 10, 2)
        min_samples_leaf = st.sidebar.slider("Min samples leaf", 1, 10, 1)

        # Model selection
        if target_type == "Categorical":
            model = DecisionTreeClassifier(
                criterion=criterion_cls,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        else:
            model = DecisionTreeRegressor(
                criterion=criterion_reg,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )

        # 6. Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 7. Evaluation
        st.subheader("📊 Model Evaluation")

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if target_type == "Categorical":
            from sklearn.metrics import accuracy_score

            st.write(f"🔵 **Training Accuracy:** {accuracy_score(y_train, y_train_pred):.4f}")
            st.write(f"🟢 **Test Accuracy:** {accuracy_score(y_test, y_test_pred):.4f}")

            st.text("📌 Confusion Matrix (Test Set)")
            cm = confusion_matrix(y_test, y_test_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.text("📌 Classification Report (Test Set)")
            st.dataframe(pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).transpose())

        else:
            st.write(f"🔵 **Training R² Score:** {r2_score(y_train, y_train_pred):.4f}")
            st.write(f"🟢 **Test R² Score:** {r2_score(y_test, y_test_pred):.4f}")

            st.write("📉 Mean Squared Error (Test Set):", mean_squared_error(y_test, y_test_pred))


        # 8. Tree Visualization
        st.subheader("🌳 Decision Tree Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_tree(model, feature_names=X.columns, filled=True)
        st.pyplot(fig)

        # 9. Feature Importances
        st.subheader("📌 Feature Importances")
        importances = pd.Series(model.feature_importances_, index=X.columns)
        st.bar_chart(importances.sort_values(ascending=False))

        # 10. Labeled Predictions
        st.subheader("⬇️ Download Labeled Predictions")
        labeled_data = df_clean.copy()
        labeled_data["prediction"] = model.predict(X)
        st.download_button(
            "Download as CSV",
            labeled_data.to_csv(index=False).encode("utf-8"),
            file_name="labeled_output.csv",
            mime="text/csv"
        )

        # 11. Export Trained Model
        st.subheader("📦 Export Trained Model")
        model_bytes = BytesIO()
        pickle.dump(model, model_bytes)
        st.download_button(
            label="Download Model (.pkl)",
            data=model_bytes.getvalue(),
            file_name="decision_tree_model.pkl",
            mime="application/octet-stream"
        )
