import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_breast_cancer
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pickle
import io
import time

# Set Streamlit page config
st.set_page_config(page_title="SVM Classifier App", layout="wide")
st.title("🔍 Support Vector Machine (SVM) Classifier")

# File uploader or use default dataset
st.sidebar.header("1. Load Data")
use_default = st.sidebar.checkbox("Use built-in Breast Cancer Dataset", help="Load sklearn's breast cancer classification dataset")

if use_default:
    data = load_breast_cancer(as_frame=True)
    df = data.frame
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Upload your dataset in CSV format")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file or select default dataset")
        st.stop()

# Sidebar variable selection
st.sidebar.header("2. Variable Selection")
target = st.sidebar.selectbox("Select Target Variable", options=df.columns, help="Choose the column you want to predict")
features = st.sidebar.multiselect("Select Feature Columns", options=[col for col in df.columns if col != target], help="Choose input features")

if not features:
    st.warning("Please select at least one feature column")
    st.stop()

# Encode target if needed
if df[target].dtype == 'object' or len(df[target].unique()) > 2:
    df[target] = LabelEncoder().fit_transform(df[target])

X = df[features]
y = df[target]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar kernel + test/train split + hyperparameters
st.sidebar.header("3. Model Settings")
kernel = st.sidebar.selectbox("Select Kernel", options=['linear', 'rbf', 'poly', 'sigmoid'], help="Kernel function to use in the algorithm")
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=30, help="Proportion of dataset used for testing") / 100
C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0, help="Penalty parameter of the error term")
gamma = st.sidebar.selectbox("Gamma", options=['scale', 'auto'], help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'")
degree = st.sidebar.slider("Degree (for poly)", 2, 5, 3, help="Degree of polynomial kernel")

params = {'C': [C], 'kernel': [kernel], 'gamma': [gamma]}
if kernel == 'poly':
    params['degree'] = [degree]

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

run_model = st.button("🚀 Run SVM Model")

# Tabs for layout
tabs = st.tabs(["📊 EDA", "📈 Modeling", "📤 Download"])

# EDA Tab
with tabs[0]:
    st.header("Exploratory Data Analysis")

    # Class balance check
    class_counts = y.value_counts()
    if (class_counts.min() / class_counts.max()) < 0.5:
        st.warning("⚠️ Class imbalance detected. This may affect model performance.")
    fig_class = px.bar(class_counts, labels={"index": "Class", "value": "Count"}, title="Target Class Distribution")
    st.plotly_chart(fig_class, use_container_width=True)

    # Pairplot using Plotly scatter matrix
    if len(features) <= 6:
        fig_pair = px.scatter_matrix(df, dimensions=features, color=target, title="Scatter Matrix")
        st.plotly_chart(fig_pair, use_container_width=True)
    else:
        st.info("Select 6 or fewer features for scatter matrix to display")

    # Correlation heatmap
    corr = df[features + [target]].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Feature Distribution")
    selected_feat = st.selectbox("Select Feature to Visualize", options=features)
    fig_feat = px.histogram(df, x=selected_feat, color=target, marginal="box", nbins=30, title=f"Distribution of {selected_feat}")
    st.plotly_chart(fig_feat, use_container_width=True)

# Modeling Tab
with tabs[1]:
    if run_model:
        start_time = time.time()
        svc = SVC(probability=True)
        grid = GridSearchCV(svc, params, cv=5)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        duration = time.time() - start_time

        st.success(f"Model trained in {duration:.2f} seconds")
        st.header("Model Evaluation")

        # Classification results
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        st.subheader("Train Classification Report")
        st.text(classification_report(y_train, y_train_pred))
        st.subheader("Test Classification Report")
        st.text(classification_report(y_test, y_test_pred))

        # Confusion matrices
        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)
        st.plotly_chart(ff.create_annotated_heatmap(z=cm_train, colorscale='Blues'), use_container_width=True)
        st.plotly_chart(ff.create_annotated_heatmap(z=cm_test, colorscale='Blues'), use_container_width=True)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc:.2f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

        # SHAP Explanation
        st.subheader("SHAP Explainability")
        try:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_test[:50])
            fig_shap, ax = plt.subplots()
            shap.summary_plot(shap_values[1], X_test[:50], feature_names=features, show=False)
            st.pyplot(fig_shap)
        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")

# Download Tab
with tabs[2]:
    if run_model:
        st.header("Download Outputs")

        # Model
        output_buffer = io.BytesIO()
        pickle.dump(model, output_buffer)
        st.download_button("Download Trained Model (.pkl)", output_buffer.getvalue(), file_name="svm_model.pkl")

        # Labeled Data
        csv_out = df.copy()
        csv_out['Prediction'] = model.predict(X_scaled)
        csv = csv_out.to_csv(index=False)
        st.download_button("Download Labeled Data (.csv)", csv, file_name="labeled_data.csv")
