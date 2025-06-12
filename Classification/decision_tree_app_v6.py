import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from io import BytesIO

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, r2_score, roc_curve, auc
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import shap

st.title("🌳 Decision Tree Classifier & Regressor App")

# 1. Load dataset
st.sidebar.header("1. Load Data")
data_source = st.sidebar.radio("Choose data source:", ["Upload CSV", "Use Iris Dataset"])

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
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

    # 2. Select features and target
    st.sidebar.header("2. Configure Variables")
    target_column = st.sidebar.selectbox("Select target variable", all_columns)
    feature_columns = st.sidebar.multiselect(
        "Select feature columns", 
        [col for col in all_columns if col != target_column],
        default=[col for col in all_columns if col != target_column]
    )

    if feature_columns:
        target_type = st.sidebar.selectbox("Target Variable Type", ["Categorical", "Continuous"])
        st.sidebar.markdown("### Feature Types")
        categorical_features = st.sidebar.multiselect("Categorical features", feature_columns)
        continuous_features = [col for col in feature_columns if col not in categorical_features]
        encoding_strategy = st.sidebar.radio("Encoding strategy", ["One-Hot Encoding", "Label Encoding"])

        # 3. Resampling method
        st.sidebar.header("3. Class Imbalance Handling")
        resampling_method = st.sidebar.selectbox("Resampling method", ["None", "SMOTE (oversample)", "Random undersample"])

        # Prepare dataset
        df_clean = df[feature_columns + [target_column]].copy()
        df_encoded = df_clean.copy()

        if encoding_strategy == "One-Hot Encoding":
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_features)
        else:
            le_dict = {}
            for col in categorical_features:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                le_dict[col] = le

        if target_type == "Categorical":
            target_encoder = LabelEncoder()
            df_encoded[target_column] = target_encoder.fit_transform(df_encoded[target_column])

        X = df_encoded.drop(columns=[target_column])
        y = df_encoded[target_column]

        # 4. Apply Resampling
        if target_type == "Categorical" and resampling_method != "None":
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

        # 6. Model Configuration
        st.sidebar.header("5. Model Parameters")
        criterion_cls = st.sidebar.selectbox("Criterion (classification)", ["gini", "entropy"])
        criterion_reg = st.sidebar.selectbox("Criterion (regression)", ["squared_error", "friedman_mse", "absolute_error"])
        max_depth = st.sidebar.slider("Max depth", 1, 30, 5)
        min_samples_split = st.sidebar.slider("Min samples split", 2, 10, 2)
        min_samples_leaf = st.sidebar.slider("Min samples leaf", 1, 10, 1)

        # 7. Train Model
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

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 8. Evaluation
        st.subheader("📊 Model Evaluation")
        if target_type == "Categorical":
            st.write(f"🔵 Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
            st.write(f"🟢 Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

            st.text("📌 Confusion Matrix (Test Set)")
            cm = confusion_matrix(y_test, y_test_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.text("📌 Classification Report (Test Set)")
            st.dataframe(pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).transpose())

        else:
            st.write(f"🔵 Training R² Score: {r2_score(y_train, y_train_pred):.4f}")
            st.write(f"🟢 Test R² Score: {r2_score(y_test, y_test_pred):.4f}")
            st.write("📉 Mean Squared Error (Test Set):", mean_squared_error(y_test, y_test_pred))

        # 9. Learning Curve
        st.subheader("📈 Learning Curve")
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5,
            scoring='accuracy' if target_type == "Categorical" else 'r2',
            train_sizes=np.linspace(0.1, 1.0, 5),
            shuffle=True, random_state=42
        )

        fig, ax = plt.subplots()
        ax.plot(train_sizes, np.mean(train_scores, axis=1), label="Training Score", marker='o')
        ax.plot(train_sizes, np.mean(test_scores, axis=1), label="Cross-validation Score", marker='o')
        ax.set_xlabel("Training Size")
        ax.set_ylabel("Score")
        ax.set_title("Learning Curve")
        ax.legend()
        st.pyplot(fig)

        # 10. ROC/AUC (binary classification)
        if target_type == "Categorical" and len(np.unique(y)) == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            st.subheader("📉 ROC Curve (Test Set)")
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig)

        # 11. Tree Visualization
        st.subheader("🌳 Decision Tree Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_tree(model, feature_names=X.columns, filled=True)
        st.pyplot(fig)

        # 12. Feature Importances
        st.subheader("📌 Feature Importances")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        fig, ax = plt.subplots()
        importances.plot(kind='bar', ax=ax)
        ax.set_title("Feature Importances")
        st.pyplot(fig)

        st.markdown("### 📊 Ranked Feature Table")
        st.dataframe(importances.reset_index().rename(columns={"index": "Feature", 0: "Importance"}))

        # 13. SHAP Explainability
        
        st.subheader("🧠 SHAP Explainability: Use for regression!")

        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        # Determine if classification or regression and handle appropriately
        is_classifier = target_type == "Categorical" and hasattr(model, "predict_proba")

        if is_classifier and shap_values.values.ndim == 3:
            # For classification, select class 1 SHAP values
            st.info("Using SHAP values for class 1 (index 1) in binary classification.")
            shap_values = shap_values[..., 1]

        # Summary Plot (Beeswarm)
        st.markdown("#### 📌 SHAP Summary Plot (Beeswarm)")
        fig = shap.plots.beeswarm(shap_values, max_display=10, show=False)
        st.pyplot(bbox_inches='tight', dpi=300, clear_figure=True)

        # Summary Plot (Bar)
        st.markdown("#### 📌 SHAP Summary Plot (Bar)")
        fig = shap.plots.bar(shap_values, max_display=10, show=False)
        st.pyplot(bbox_inches='tight', dpi=300, clear_figure=True)

        #st.components.v1.html(shap_html, height=300)





        # 14. Download labeled predictions
        st.subheader("⬇️ Download Labeled Predictions")
        labeled_data = df_clean.copy()
        labeled_data["prediction"] = model.predict(X)
        st.download_button(
            "Download as CSV",
            labeled_data.to_csv(index=False).encode("utf-8"),
            file_name="labeled_output.csv",
            mime="text/csv"
        )

        # 15. Download trained model
        st.subheader("📦 Export Trained Model")
        model_bytes = BytesIO()
        pickle.dump(model, model_bytes)
        st.download_button(
            label="Download Model (.pkl)",
            data=model_bytes.getvalue(),
            file_name="decision_tree_model.pkl",
            mime="application/octet-stream"
        )

