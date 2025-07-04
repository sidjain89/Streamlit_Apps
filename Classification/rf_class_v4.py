import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

st.set_page_config(page_title="Random Forest Classifier", layout="wide")
st.title("🌲 Random Forest Classification App")

# Upload data
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    with st.sidebar:
        st.header("Feature Selection")
        all_columns = df.columns.tolist()
        default_target = df.columns[-1]
        target_column = st.selectbox("Select target column", all_columns, index=all_columns.index(default_target))

        feature_options = [col for col in all_columns if col != target_column]
        categorical_features = st.multiselect("Select categorical features", feature_options)
        continuous_features = st.multiselect("Select continuous (numeric) features", [col for col in feature_options if col not in categorical_features])

        selected_features = categorical_features + continuous_features

        test_size_ratio = st.slider("Select test set size (as a proportion)", 0.1, 0.5, 0.2, 0.05)

    if selected_features:
        df_encoded = df.copy()
        for col in categorical_features:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        X = df_encoded[selected_features]
        y = df_encoded[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42)

        tab1, tab2, tab3 = st.tabs(["🔍 EDA", "🧠 Model", "📦 Download"])

        with tab1:
            st.subheader("Exploratory Data Analysis")
            with st.expander("🔍 Custom EDA Visualizations", expanded=False):
                if selected_features and target_column:
                    eda_options = st.multiselect(
                        "Select EDA plots to show",
                        ["Correlation Heatmap", "Boxplot by Target", "Histogram", "Class Distribution"],
                        default=["Correlation Heatmap"]
                    )

                    if "Correlation Heatmap" in eda_options:
                        st.markdown("### Correlation Heatmap")
                        numeric_df = df_encoded[continuous_features]
                        if numeric_df.shape[1] >= 2:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                            st.pyplot(fig)
                        else:
                            st.warning("Need at least two numeric features for correlation heatmap.")

                    if "Boxplot by Target" in eda_options:
                        st.markdown("### Boxplot by Target")
                        numeric_cols = continuous_features
                        if numeric_cols:
                            feature_for_box = st.selectbox("Select numeric feature for boxplot", numeric_cols, key="boxplot_feature")
                            fig, ax = plt.subplots()
                            sns.boxplot(x=df[target_column], y=df[feature_for_box], ax=ax)
                            st.pyplot(fig)
                        else:
                            st.warning("No numeric features available.")

                    if "Histogram" in eda_options:
                        st.markdown("### Histogram")
                        feature_for_hist = st.selectbox("Select numeric feature for histogram", continuous_features, key="hist_feature")
                        fig, ax = plt.subplots()
                        sns.histplot(df[feature_for_hist], kde=True, ax=ax)
                        st.pyplot(fig)

                    if "Class Distribution" in eda_options:
                        st.markdown("### Class Distribution")
                        fig, ax = plt.subplots()
                        sns.countplot(x=df[target_column], ax=ax)
                        st.pyplot(fig)

        with tab2:
            st.subheader("Train Random Forest Classifier")

            st.sidebar.header("Model Hyperparameters")
            n_estimators = st.sidebar.slider("Number of trees", 10, 300, 100, 10)
            max_depth = st.sidebar.slider("Max depth", 1, 20, 5)
            max_features = st.sidebar.selectbox("Max features", ["sqrt", "log2", None])

            use_optuna = st.checkbox("Use Optuna to tune hyperparameters")

            if "model_trained" not in st.session_state:
                st.session_state["model_trained"] = False

            if st.button("Train Model"):
                if use_optuna:
                    def objective(trial):
                        params = {
                            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                            'max_depth': trial.suggest_int('max_depth', 2, 20),
                            'max_features': trial.suggest_categorical('max_features', ["sqrt", "log2", None])
                        }
                        model = RandomForestClassifier(**params, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        return roc_auc_score(y_test, y_pred)

                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=20)
                    st.write("Best trial:", study.best_trial.params)

                    best_params = study.best_trial.params
                    model = RandomForestClassifier(**best_params, random_state=42)
                else:
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        max_features=max_features,
                        random_state=42
                    )

                model.fit(X_train, y_train)
                st.success("Model trained successfully!")

                st.session_state["model"] = model
                st.session_state["model_trained"] = True
                st.session_state["X_test"] = X_test
                st.session_state["X_train"] = X_train
                st.session_state["y_test"] = y_test
                st.session_state["y_train"] = y_train

            if st.session_state["model_trained"]:
                model = st.session_state["model"]
                X_test = st.session_state["X_test"]
                X_train = st.session_state["X_train"]
                y_test = st.session_state["y_test"]
                y_train = st.session_state["y_train"]

                st.subheader("Model Evaluation")
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                st.markdown("**Train Classification Report:**")
                st.text(classification_report(y_train, y_train_pred))

                st.markdown("**Test Classification Report:**")
                st.text(classification_report(y_test, y_test_pred))

                st.markdown("**Confusion Matrix (Test):**")
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                st.markdown("**Feature Importance:**")
                feat_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig, ax = plt.subplots()
                sns.barplot(x=feat_importance, y=feat_importance.index, ax=ax)
                st.pyplot(fig)

                st.subheader("SHAP Visualizations")
                shap_plot_type = st.selectbox("Select SHAP plot type", ["summary", "bar", "dependence", "bee swarm"], key="shap_plot_type")

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)

                if shap_plot_type == "summary":
                    fig = plt.figure()
                    shap.summary_plot(shap_values, X_test, show=False)
                    st.pyplot(fig)
                elif shap_plot_type == "bar":
                    fig = plt.figure()
                    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                    st.pyplot(fig)
                elif shap_plot_type == "dependence":
                    feature_for_dep = st.selectbox("Select feature for dependence plot", X_test.columns, key="dep_plot_feature")
                    fig = plt.figure()
                    shap.dependence_plot(feature_for_dep, shap_values[1], X_test, show=False)
                    st.pyplot(fig)
                elif shap_plot_type == "bee swarm":
                    fig = plt.figure()
                    shap.plots.beeswarm(shap.Explanation(values=shap_values[1], data=X_test, feature_names=X_test.columns), show=False)
                    st.pyplot(fig)

        with tab3:
            st.subheader("📦 Download Trained Model and Data")

            if "model" in st.session_state:
                model_filename = "random_forest_model.pkl"
                data_filename = "labelled_data.csv"

                with open(model_filename, "wb") as f:
                    pickle.dump(st.session_state["model"], f)

                df_encoded.to_csv(data_filename, index=False)

                with open(model_filename, "rb") as f:
                    st.download_button("Download Model (Pickle)", f, file_name=model_filename)

                with open(data_filename, "rb") as f:
                    st.download_button("Download Encoded Data (CSV)", f, file_name=data_filename)

else:
    st.warning("Please upload a CSV file to begin.")
