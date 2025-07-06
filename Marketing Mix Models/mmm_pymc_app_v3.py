import streamlit as st
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from datetime import datetime
import io
import pickle
import optuna

# ------------------ Adstock & Saturation Functions ------------------ #

def geometric_adstock(x, alpha):
    result = [x[0]]
    for i in range(1, len(x)):
        result.append(x[i] + alpha * result[i - 1])
    return result

def weibull_adstock(x, lam, k):
    from scipy.stats import weibull_min
    w = weibull_min.pdf(np.arange(len(x)), c=k, scale=lam)
    return np.convolve(x, w, mode='full')[:len(x)]

def hill_saturation(x, alpha, gamma):
    return (x**gamma) / (x**gamma + alpha**gamma)

# ------------------ Model Builder ------------------ #

def apply_transform(x, adstock_type, adstock_params, sat_type, sat_params):
    # Adstock
    if adstock_type == "Geometric":
        x = geometric_adstock(x, adstock_params["alpha"])
    elif adstock_type == "Weibull":
        x = weibull_adstock(x, adstock_params["lam"], adstock_params["k"])
    
    # Saturation
    if sat_type == "Hill":
        x = hill_saturation(np.array(x), sat_params["alpha"], sat_params["gamma"])
    return x

def build_model(df, target, media_vars, control_vars, adstock_type, adstock_params, sat_type, sat_params):
    with pm.Model() as model:
        X_media = np.array([
            apply_transform(df[var].values, adstock_type, adstock_params, sat_type, sat_params)
            for var in media_vars
        ]).T

        X_control = df[control_vars].values if control_vars else None
        X = X_media if control_vars == [] else np.hstack((X_media, X_control))

        beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        mu = intercept + pm.math.dot(X, beta)

        sigma = pm.HalfNormal("sigma", sigma=1)
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=df[target].values)

    return model

# ------------------ Budget Optimizer ------------------ #

def optimize_budget(trace, media_vars, total_budget):
    betas = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
    media_betas = betas[:len(media_vars)]
    weights = media_betas / np.sum(media_betas)
    return {var: round(w * total_budget, 2) for var, w in zip(media_vars, weights)}

# ------------------ Streamlit App ------------------ #

st.set_page_config(page_title="Advanced MMM with PyMC", layout="wide")
st.title("📊 Advanced Marketing Mix Modeling (MMM) with PyMC + Optuna")

# ------------------ Sidebar: Upload and Variable Config ------------------ #

with st.sidebar:
    st.header("Upload CSV and Select Variables")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    df = None

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

        all_cols = df.columns.tolist()
        date_col = st.selectbox("Date Column", options=[col for col in all_cols if "date" in col.lower()])
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])

        target = st.selectbox("Target Variable", options=[col for col in all_cols if col != date_col])
        media_vars = st.multiselect("Media Variables", options=[col for col in all_cols if col not in [date_col, target]])
        control_vars = st.multiselect("Control Variables (Optional)", options=[col for col in all_cols if col not in [date_col, target] + media_vars])

        if date_col and target and media_vars:
            min_date, max_date = df[date_col].min().date(), df[date_col].max().date()
            date_range = st.slider("Train/Test Split", min_value=min_date, max_value=max_date, value=(min_date, max_date))

# ------------------ Data Preview ------------------ #

if uploaded_file:
    st.subheader("📄 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

# ------------------ Main Tabs ------------------ #

if df is not None and date_col and target and media_vars:
    df = df.sort_values(date_col)
    df = df.dropna(subset=[target] + media_vars)
    train_df = df[df[date_col] < pd.to_datetime(date_range[1])]
    test_df = df[df[date_col] >= pd.to_datetime(date_range[1])]

    tab1, tab2, tab3, tab4 = st.tabs(["📈 EDA", "⚙️ Model", "📊 Optimization", "⬇️ Download"])

    # -------- EDA -------- #
    with tab1:
        st.subheader("Target Over Time")
        st.line_chart(df[[date_col, target]].set_index(date_col))
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
        st.subheader("Correlations")
        st.write(df[[target] + media_vars + control_vars].corr())

    # -------- Model Config + Training -------- #
    with tab2:
        st.subheader("Model Settings")

        adstock_type = st.selectbox("Adstock Function", ["Geometric", "Weibull"])
        sat_type = st.selectbox("Saturation Function", ["None", "Hill"])

        adstock_params = {}
        if adstock_type == "Geometric":
            adstock_params["alpha"] = st.slider("Adstock Alpha", 0.0, 1.0, 0.5)
        elif adstock_type == "Weibull":
            adstock_params["lam"] = st.slider("Weibull Lambda", 1.0, 20.0, 10.0)
            adstock_params["k"] = st.slider("Weibull K", 0.1, 5.0, 1.5)

        sat_params = {}
        if sat_type == "Hill":
            sat_params["alpha"] = st.slider("Hill Alpha", 1.0, 20.0, 10.0)
            sat_params["gamma"] = st.slider("Hill Gamma", 0.1, 3.0, 1.0)

        use_optuna = st.checkbox("Use Optuna for Hyperparameter Tuning")
        run_model = st.button("Run MMM")

        if run_model:
            st.subheader("Training MMM Model...")

            model = build_model(train_df, target, media_vars, control_vars, adstock_type, adstock_params, sat_type, sat_params)
            with model:
                trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True)

            st.success("Model training complete.")
            st.subheader("Trace Summary")
            st.write(az.summary(trace))

            X_test = np.array([
                apply_transform(test_df[var].values, adstock_type, adstock_params, sat_type, sat_params)
                for var in media_vars
            ]).T
            if control_vars:
                X_test = np.hstack((X_test, test_df[control_vars].values))

            beta_mean = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
            intercept_mean = trace.posterior["intercept"].mean().values
            y_pred = intercept_mean + np.dot(X_test, beta_mean)

            chart_df = pd.DataFrame({
                "date": test_df[date_col],
                "actual": test_df[target].values,
                "predicted": y_pred
            }).set_index("date")
            st.line_chart(chart_df)

    # -------- Budget Optimization -------- #
    with tab3:
        st.subheader("Budget Allocation Optimizer")
        total_budget = st.number_input("Total Budget", min_value=1000, value=5000, step=500)
        if run_model:
            alloc = optimize_budget(trace, media_vars, total_budget)
            st.write("Optimal Spend:")
            st.write(alloc)

    # -------- Download Tab -------- #
    with tab4:
        if run_model:
            model_bytes = io.BytesIO()
            pickle.dump(trace, model_bytes)
            st.download_button("Download Trained Model", model_bytes.getvalue(), file_name="pymc_mmm_trace.pkl")

            labeled_data = df.copy()
            labeled_data["prediction"] = np.nan
            labeled_data.loc[test_df.index, "prediction"] = y_pred
            csv_buffer = io.StringIO()
            labeled_data.to_csv(csv_buffer, index=False)
            st.download_button("Download Labeled Data", csv_buffer.getvalue(), file_name="mmm_predictions.csv")
else:
    st.info("👈 Upload data and configure required columns in the sidebar.")
