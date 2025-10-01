import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="LightGBM Classification", layout="wide")

st.title("🌟 LightGBM Classification App")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to:", ["Upload Data", "Preprocessing", "Model Training", "Model Evaluation"])

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}

# Upload Data Section
if options == "Upload Data":
    st.header("📂 Upload Your Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.data.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Shape:**", st.session_state.data.shape)
        with col2:
            st.write("**Missing Values:**", st.session_state.data.isnull().sum().sum())
        
        st.subheader("Dataset Info")
        buffer = io.StringIO()
        st.session_state.data.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.subheader("Statistical Summary")
        st.dataframe(st.session_state.data.describe())

# Preprocessing Section
elif options == "Preprocessing":
    st.header("🔧 Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("Please upload data first!")
    else:
        df = st.session_state.data.copy()
        
        st.subheader("Select Target Variable")
        target_column = st.selectbox("Choose target column:", df.columns)
        
        st.subheader("Handle Missing Values")
        missing_option = st.radio("Choose method:", 
                                  ["Drop rows with missing values", 
                                   "Fill with mean (numeric)", 
                                   "Fill with mode (categorical)"])
        
        if missing_option == "Drop rows with missing values":
            df = df.dropna()
            st.info(f"Dropped rows. New shape: {df.shape}")
        elif missing_option == "Fill with mean (numeric)":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            st.info("Filled numeric columns with mean")
        else:
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
            st.info("Filled columns with mode")
        
        st.subheader("Encode Categorical Variables")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        if categorical_cols:
            st.write("Categorical columns found:", categorical_cols)
            encode_option = st.checkbox("Apply Label Encoding to categorical features")
            
            if encode_option:
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    st.session_state.label_encoders[col] = le
                st.success("Label encoding applied!")
        
        # Encode target if categorical
        if df[target_column].dtype == 'object':
            le_target = LabelEncoder()
            df[target_column] = le_target.fit_transform(df[target_column])
            st.session_state.label_encoders['target'] = le_target
            st.info(f"Target variable encoded. Classes: {le_target.classes_}")
        
        st.subheader("Train-Test Split")
        test_size = st.slider("Test size ratio:", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state:", 0, 1000, 42)
        
        if st.button("Split Data"):
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            st.session_state.X_train, st.session_state.X_test, \
            st.session_state.y_train, st.session_state.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            st.success(f"Data split successfully!")
            st.write(f"Training set size: {st.session_state.X_train.shape}")
            st.write(f"Test set size: {st.session_state.X_test.shape}")

# Model Training Section
elif options == "Model Training":
    st.header("🤖 Train LightGBM Model")
    
    if st.session_state.X_train is None:
        st.warning("Please complete preprocessing first!")
    else:
        st.subheader("Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_leaves = st.slider("Number of leaves:", 10, 200, 31)
            max_depth = st.slider("Max depth:", -1, 50, -1)
            learning_rate = st.number_input("Learning rate:", 0.001, 1.0, 0.1, 0.01)
            n_estimators = st.slider("Number of estimators:", 10, 1000, 100)
        
        with col2:
            min_child_samples = st.slider("Min child samples:", 1, 100, 20)
            subsample = st.slider("Subsample ratio:", 0.1, 1.0, 1.0, 0.1)
            colsample_bytree = st.slider("Column sample by tree:", 0.1, 1.0, 1.0, 0.1)
            reg_alpha = st.number_input("L1 regularization (alpha):", 0.0, 10.0, 0.0, 0.1)
            reg_lambda = st.number_input("L2 regularization (lambda):", 0.0, 10.0, 0.0, 0.1)
        
        boosting_type = st.selectbox("Boosting type:", ["gbdt", "dart", "goss"])
        objective = st.selectbox("Objective:", ["binary", "multiclass"])
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                params = {
                    'num_leaves': num_leaves,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'n_estimators': n_estimators,
                    'min_child_samples': min_child_samples,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree,
                    'reg_alpha': reg_alpha,
                    'reg_lambda': reg_lambda,
                    'boosting_type': boosting_type,
                    'objective': objective,
                    'random_state': 42,
                    'verbose': -1
                }
                
                if objective == 'multiclass':
                    params['num_class'] = len(np.unique(st.session_state.y_train))
                
                st.session_state.model = lgb.LGBMClassifier(**params)
                st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
                
                st.success("Model trained successfully!")
                
                # Feature importance
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'feature': st.session_state.X_train.columns,
                    'importance': st.session_state.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df.head(20), x='importance', y='feature', ax=ax)
                ax.set_title("Top 20 Feature Importances")
                st.pyplot(fig)

# Model Evaluation Section
elif options == "Model Evaluation":
    st.header("📊 Model Evaluation")
    
    if st.session_state.model is None:
        st.warning("Please train the model first!")
    else:
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        y_pred_proba = st.session_state.model.predict_proba(st.session_state.X_test)
        
        st.subheader("Model Performance Metrics")
        
        accuracy = accuracy_score(st.session_state.y_test, y_pred)
        st.metric("Accuracy", f"{accuracy:.4f}")
        
        st.subheader("Classification Report")
        report = classification_report(st.session_state.y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        # ROC Curve (for binary classification)
        if len(np.unique(st.session_state.y_train)) == 2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(st.session_state.y_test, y_pred_proba[:, 1])
            auc = roc_auc_score(st.session_state.y_test, y_pred_proba[:, 1])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        # Prediction Distribution
        st.subheader("Prediction Distribution")
        pred_df = pd.DataFrame({
            'Actual': st.session_state.y_test,
            'Predicted': y_pred
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Actual Distribution**")
            st.bar_chart(pred_df['Actual'].value_counts())
        with col2:
            st.write("**Predicted Distribution**")
            st.bar_chart(pred_df['Predicted'].value_counts())

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit & LightGBM")