import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="EDA App", layout="wide")
st.title("📊 Exploratory Data Analysis (EDA) App")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

target_col = None
target_type = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    target_col = st.sidebar.selectbox("Select Target Variable", all_columns)
    target_type = st.sidebar.radio("Target Variable Type", ["Categorical", "Continuous"])

    available_features = [col for col in all_columns if col != target_col]
    categorical_vars = st.sidebar.multiselect("Select Categorical Features", available_features)
    available_for_continuous = [col for col in available_features if col not in categorical_vars]
    continuous_vars = st.sidebar.multiselect("Select Continuous Features", available_for_continuous)

    tab_cont, tab_cat, tab_multi, tab_summary, tab_pair = st.tabs([
        "Continuous EDA", "Categorical EDA", "Multivariate Analysis", "Summary Statistics", "Pair Plots"])

    with tab_cont:
        st.header("📈 Continuous Variables EDA")
        for col in continuous_vars:
            with st.expander(f"{col} - Histogram"):
                fig = px.histogram(df, x=col, marginal="box", nbins=30, title=f"Histogram of {col}")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander(f"{col} - KDE + Boxplot"):
                fig = px.box(df, y=col, points="all", title=f"Boxplot of {col}")
                st.plotly_chart(fig, use_container_width=True)

    with tab_cat:
        st.header("📊 Categorical Variables EDA")
        for col in categorical_vars:
            with st.expander(f"{col} - Bar Chart"):
                count_df = df[col].value_counts().reset_index()
                count_df.columns = [col, 'count']
                fig = px.bar(count_df, x=col, y='count', title=f"Bar Chart of {col}")
                st.plotly_chart(fig, use_container_width=True)

    with tab_multi:
        st.header("🔀 Multivariate Analysis")

        if target_col and target_type == "Categorical" and len(continuous_vars) > 0:
            st.subheader(f"Boxplots of Continuous Variables by Target: {target_col}")
            for col in continuous_vars:
                with st.expander(f"{col} by {target_col}"):
                    fig = px.box(df, x=target_col, y=col, points="all", title=f"{col} by {target_col}")
                    st.plotly_chart(fig, use_container_width=True)

        if target_col and target_type == "Continuous" and len(categorical_vars) > 0:
            st.subheader(f"Violin/Strip Plots of Target by Categorical Features")
            for col in categorical_vars:
                with st.expander(f"{target_col} by {col}"):
                    fig = px.violin(df, x=col, y=target_col, box=True, points="all", title=f"{target_col} by {col}")
                    st.plotly_chart(fig, use_container_width=True)

        if len(continuous_vars) >= 2:
            st.subheader("Scatter Plots Between Continuous Variables")
            var1 = st.selectbox("X-axis", continuous_vars, key="scatter_x")
            var2 = st.selectbox("Y-axis", [col for col in continuous_vars if col != var1], key="scatter_y")
            fig = px.scatter(df, x=var1, y=var2, color=target_col if target_col else None, title=f"{var1} vs {var2}")
            st.plotly_chart(fig, use_container_width=True)

    with tab_summary:
        st.header("📋 Summary Statistics")

        def to_csv(df_):
            return df_.to_csv(index=True).encode('utf-8')

        def to_excel(df_):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_.to_excel(writer, index=True, sheet_name='Summary')
                writer.close()
            processed_data = output.getvalue()
            return processed_data

        if len(continuous_vars) > 0:
            st.subheader("Continuous Variables")
            cont_desc = df[continuous_vars].describe().T
            cont_desc['missing_count'] = df[continuous_vars].isna().sum()
            cont_desc['missing_pct'] = df[continuous_vars].isna().mean().mul(100).round(2)
            st.dataframe(cont_desc)

            col1, col2 = st.columns(2)
            with col1:
                csv = to_csv(cont_desc)
                st.download_button("Download Continuous Stats as CSV", data=csv, file_name='continuous_summary.csv', mime='text/csv')
            with col2:
                excel = to_excel(cont_desc)
                st.download_button("Download Continuous Stats as Excel", data=excel, file_name='continuous_summary.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            if target_col and target_type == "Categorical":
                st.subheader(f"Grouped Continuous Stats by Target: {target_col}")
                grouped_cont = df.groupby(target_col)[continuous_vars].describe().T
                st.dataframe(grouped_cont)

                col3, col4 = st.columns(2)
                with col3:
                    csv_g = to_csv(grouped_cont)
                    st.download_button(f"Download Grouped Continuous Stats CSV", data=csv_g, file_name='grouped_continuous_summary.csv', mime='text/csv')
                with col4:
                    excel_g = to_excel(grouped_cont)
                    st.download_button(f"Download Grouped Continuous Stats Excel", data=excel_g, file_name='grouped_continuous_summary.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        if len(categorical_vars) > 0:
            st.subheader("Categorical Variables")
            cat_summary = pd.DataFrame({
                'missing_count': df[categorical_vars].isna().sum(),
                'missing_pct': df[categorical_vars].isna().mean().mul(100).round(2),
                'unique_count': df[categorical_vars].nunique()
            })
            st.dataframe(cat_summary)

            col5, col6 = st.columns(2)
            with col5:
                csv_cat = to_csv(cat_summary)
                st.download_button("Download Categorical Summary CSV", data=csv_cat, file_name='categorical_summary.csv', mime='text/csv')
            with col6:
                excel_cat = to_excel(cat_summary)
                st.download_button("Download Categorical Summary Excel", data=excel_cat, file_name='categorical_summary.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            if target_col and target_type == "Continuous":
                st.subheader(f"Grouped Categorical Stats by Target: {target_col}")
                for cat_col in categorical_vars:
                    st.markdown(f"**{cat_col}**")
                    grouped_cat = df.groupby(cat_col)[target_col].describe()
                    st.dataframe(grouped_cat)

                    csv_gcat = to_csv(grouped_cat)
                    excel_gcat = to_excel(grouped_cat)
                    col7, col8 = st.columns(2)
                    with col7:
                        st.download_button(f"Download {cat_col} grouped stats CSV", data=csv_gcat, file_name=f'{cat_col}_grouped_categorical_summary.csv', mime='text/csv')
                    with col8:
                        st.download_button(f"Download {cat_col} grouped stats Excel", data=excel_gcat, file_name=f'{cat_col}_grouped_categorical_summary.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        st.subheader("Overall Missing Values")
        missing_df = pd.DataFrame({
            'missing_count': df.isna().sum(),
            'missing_pct': df.isna().mean().mul(100).round(2)
        }).sort_values('missing_pct', ascending=False)
        st.dataframe(missing_df)

        csv_missing = to_csv(missing_df)
        excel_missing = to_excel(missing_df)

        col9, col10 = st.columns(2)
        with col9:
            st.download_button("Download Missing Values CSV", data=csv_missing, file_name='missing_values.csv', mime='text/csv')
        with col10:
            st.download_button("Download Missing Values Excel", data=excel_missing, file_name='missing_values.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    with tab_pair:
        st.header("🔗 Pair Plots")
        if len(continuous_vars) >= 2:
            st.markdown("Pairplot for selected continuous variables")
            fig = sns.pairplot(df[continuous_vars])
            st.pyplot(fig)
        else:
            st.warning("Please select at least 2 continuous variables to generate a pairplot.")
