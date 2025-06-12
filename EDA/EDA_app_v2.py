import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="🔍 Enhanced EDA App", layout="wide")
st.title("📊 Exploratory Data Analysis App")

# -----------------------------
# Dataset Input
# -----------------------------
option = st.selectbox("Choose data source:", ["Upload CSV", "Use sample Titanic dataset"])
if option == "Upload CSV":
    file = st.file_uploader("Upload your dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
    else:
        st.stop()
else:
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# -----------------------------
# Dynamic Filtering / Segmentation
# -----------------------------
with st.sidebar:
    st.header("🔍 Filter / Segment Data")

    df_filtered = df.copy()
    filter_cols = st.multiselect("Select columns to filter by", df.columns.tolist())

    for col in filter_cols:
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            selected_values = st.multiselect(f"Filter {col}", df[col].dropna().unique())
            if selected_values:
                df_filtered = df_filtered[df_filtered[col].isin(selected_values)]
        elif np.issubdtype(df[col].dtype, np.number):
            min_val, max_val = df[col].min(), df[col].max()
            selected_range = st.slider(f"Filter {col}", float(min_val), float(max_val), (float(min_val), float(max_val)))
            df_filtered = df_filtered[df_filtered[col].between(*selected_range)]
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            min_date, max_date = df[col].min(), df[col].max()
            selected_date = st.date_input(f"Filter {col}", (min_date, max_date))
            if isinstance(selected_date, tuple) and len(selected_date) == 2:
                df_filtered = df_filtered[df_filtered[col].between(*selected_date)]

# -----------------------------
# Variable Config
# -----------------------------
with st.sidebar:
    st.header("🛠️ Configure Variables")
    all_columns = df_filtered.columns.tolist()
    target = st.selectbox("Target variable (optional)", [None] + all_columns)
    categorical_vars = st.multiselect("Categorical variables", all_columns)
    numeric_vars = st.multiselect("Numeric variables", df_filtered.select_dtypes(include=np.number).columns.tolist())

# -----------------------------
# Preview Data
# -----------------------------
st.write("### Preview")
st.dataframe(df_filtered.head())

# -----------------------------
# Pandas Profiling
# -----------------------------
st.subheader("📑 Pandas Profiling Report")
if st.button("Generate Profiling Report"):
    with st.spinner("Generating report..."):
        profile = ProfileReport(df_filtered, title="Filtered Data Profiling", explorative=True)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            profile.to_file(f.name)
            html(open(f.name, encoding='utf-8').read(), height=800, scrolling=True)
            st.download_button("📥 Download Report", f.read(), file_name="profiling_report.html")

# -----------------------------
# Plotly Visualizations
# -----------------------------
st.subheader("📈 Interactive Charts")
chart_type = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap"])

if chart_type == "Histogram":
    col = st.selectbox("Select column", categorical_vars + numeric_vars)
    fig = px.histogram(df_filtered, x=col, color=target if target else None)
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Box Plot":
    y = st.selectbox("Y-axis", numeric_vars)
    x = st.selectbox("Group by", categorical_vars)
    fig = px.box(df_filtered, x=x, y=y, color=x)
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Scatter Plot":
    x = st.selectbox("X-axis", numeric_vars, key='x')
    y = st.selectbox("Y-axis", numeric_vars, key='y')
    color = st.selectbox("Color by", [None] + categorical_vars + numeric_vars)
    fig = px.scatter(df_filtered, x=x, y=y, color=color)
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Correlation Heatmap":
    st.write("### Correlation Heatmap")
    corr = df_filtered[numeric_vars].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# -----------------------------
# Outlier Detection
# -----------------------------
st.subheader("🚨 Outlier Visualization")
selected_outlier_col = st.selectbox("Select column for outlier detection", numeric_vars)
fig = px.box(df_filtered, y=selected_outlier_col, points="all", title=f"Outlier Detection: {selected_outlier_col}")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Feature Importance (if target selected)
# -----------------------------
if target and len(numeric_vars) > 1:
    st.subheader("⭐ Feature Importance")
    df_encoded = df_filtered.copy()

    if df_encoded[target].dtype == 'object':
        le = LabelEncoder()
        df_encoded[target] = le.fit_transform(df_encoded[target].astype(str))
        model = RandomForestClassifier(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    X = df_encoded[numeric_vars].dropna()
    y = df_encoded.loc[X.index, target]
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    
    fig = px.bar(importances, orientation='h', title="Feature Importances")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Descriptive Stats
# -----------------------------
st.subheader("📋 Descriptive Statistics")
st.dataframe(df_filtered.describe(include='all').T)
