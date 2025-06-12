import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from io import BytesIO

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import shap

st.title("🧮 Logistic Regression")

# 1. Load Dataset
st.sidebar.header("1. Load Data")
data_source = st.sidebar.radio("Choose data source:", ["Upload CSV", "Use Iris Dataset"])

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df = df[df["target"] != 2]  # Binary classification only

if 'df' in locals():
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()

    # 2. Configure Inputs
    st.sidebar.header("2. Configure Features")
    target_column = st.sidebar.selectbox("Select target variable", all_columns)
    feature_columns = st.sidebar.multiselect(
        "Select feature columns", 
        [col for col in all_columns if col != target_column],
        default=[col for col in all_columns if col != target_column]
    )

    categorical_features = st.sidebar.multiselect("Categorical features", feature_columns)
    encoding_strategy = st.sidebar.radio("Encoding strategy", ["One-Hot Encoding", "Label Encoding"])

    # 3. Resampling
    st.sidebar.header("3. Class Imbalance")
    resampling_method = st.sidebar.selectbox("Resampling method", ["None", "SMOTE (oversample)", "Random undersample"])

    df_clean = df[feature_columns + [target_column]].copy()
    df_encoded = df_clean.copy()

    # Encode categorical features
    if encoding_strategy == "One-Hot Encoding":
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_features)
    else:
        for col in categorical_features:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])

    # Encode target
    target_encoder = LabelEncoder()
    df_encoded[target_column] = target_encoder.fit_transform(df_encoded[target_column])

    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]

    # 4. Resampling
    if resampling_method != "None":
        if resampling_method == "SMOTE (oversample)":
            sm = SMOTE(random_state=42)
            X, y = sm.fit_resample(X, y)
        elif resampling_method == "Random undersample":
            rus = RandomUnderSampler(random_state=42)
            X, y = rus.fit_resample(X, y)

    # 5. Train/Test Split
    st.sidebar.header("4. Train/Test Split")
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 6. Model Config & Training
    st.sidebar.header("5. Logistic Regression Params")
    C = st.sidebar.slider("Inverse Regularization Strength (C)", 0.001, 10.0, 1.0)
    solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "saga", "newton-cg"])

    model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 6b. Display Logistic Regression Equation
    st.subheader("🧾 Logistic Regression Equation")

    coef = model.coef_[0]
    intercept = model.intercept_[0]
    terms = []

    for feature, weight in zip(X.columns, coef):
        terms.append(f"({weight:.4f} × {feature})")

    equation = f"logit(p) = {intercept:.4f} + " + " + ".join(terms)
    st.code(equation)


    # 7. Evaluation
    st.subheader("📊 Model Evaluation")
    st.write(f"🔵 Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    st.write(f"🟢 Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

    st.text("📌 Confusion Matrix (Test Set)")
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.text("📌 Classification Report (Test Set)")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).transpose())

    # 8. ROC Curve & AUC
    st.subheader("📉 ROC Curve & AUC")
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    # 9. SHAP Explainability
    st.subheader("🧠 SHAP Explainability")

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # SHAP Summary
    st.markdown("#### 📌 SHAP Summary Plot (Beeswarm)")
    fig = shap.plots.beeswarm(shap_values, max_display=10, show=False)
    st.pyplot(bbox_inches='tight', dpi=300, clear_figure=True)

    st.markdown("#### 📌 SHAP Summary Plot (Bar)")
    fig = shap.plots.bar(shap_values, max_display=10, show=False)
    st.pyplot(bbox_inches='tight', dpi=300, clear_figure=True)

    # 10. Download Predictions
    st.subheader("⬇️ Download Labeled Predictions")
    labeled_data = df_clean.copy()
    labeled_data["prediction"] = model.predict(X)
    st.download_button(
        "Download as CSV",
        labeled_data.to_csv(index=False).encode("utf-8"),
        file_name="logistic_predictions.csv",
        mime="text/csv"
    )

    # 11. Export Model
    st.subheader("📦 Export Trained Model")
    model_bytes = BytesIO()
    pickle.dump(model, model_bytes)
    st.download_button(
        label="Download Model (.pkl)",
        data=model_bytes.getvalue(),
        file_name="logistic_model.pkl",
        mime="application/octet-stream"
    )
