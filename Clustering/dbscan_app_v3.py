import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import base64
import time

# Utility Functions
def load_data(use_default=True, file=None):
    if use_default:
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        return iris.frame.drop(columns=["target"])
    else:
        return pd.read_csv(file)

def download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# EDA Functions
def run_eda(data: pd.DataFrame):
    st.subheader("📊 Exploratory Data Analysis")
    if st.checkbox("Show Summary Statistics"):
        st.write(data.describe())
    if st.checkbox("Show Pairplot"):
        numeric_cols = data.select_dtypes(include='number')
        if numeric_cols.shape[1] <= 1:
            st.warning("Need at least 2 numeric columns for pairplot.")
        else:
            fig = sns.pairplot(numeric_cols)
            st.pyplot(fig)
    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    if st.checkbox("Show Histograms"):
        numeric_cols = data.select_dtypes(include='number')
        for col in numeric_cols.columns:
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, ax=ax)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)

# Visualization Functions
def plot_2d(X, labels):
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow", s=50)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("2D Cluster Visualization")
    st.pyplot(fig)

def plot_3d_interactive(X, labels):
    df = pd.DataFrame(X[:, :3], columns=["Feature 1", "Feature 2", "Feature 3"])
    df["Cluster"] = labels.astype(str)
    fig = px.scatter_3d(df, x="Feature 1", y="Feature 2", z="Feature 3",
                        color="Cluster", symbol="Cluster",
                        title="Interactive 3D Cluster Visualization",
                        width=800, height=600)
    st.plotly_chart(fig, use_container_width=True)

def plot_k_distance(X, min_samples):
    neigh = NearestNeighbors(n_neighbors=min_samples)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = np.sort(distances[:, -1])
    fig, ax = plt.subplots()
    ax.plot(distances)
    ax.set_ylabel("k-distance")
    ax.set_xlabel("Points sorted by distance")
    ax.set_title("k-distance Graph for eps estimation")
    st.pyplot(fig)

def plot_radar(cluster_data, cluster_labels):
    from math import pi
    cluster_means = cluster_data.groupby(cluster_labels).mean()
    categories = cluster_means.columns
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i in range(len(cluster_means)):
        values = cluster_means.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, label=f"Cluster {i}")
        ax.fill(angles, values, alpha=0.1)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, rotation=45)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    st.pyplot(fig)

# Clustering Functions
def run_dbscan(data: pd.DataFrame):
    st.subheader("📌 DBSCAN Clustering")
    numeric_data = data.select_dtypes(include='number')
    if numeric_data.shape[1] < 2:
        st.warning("Not enough numeric features for clustering (need at least 2).")
        return
    X_scaled = StandardScaler().fit_transform(numeric_data)
    st.markdown("### 🔧 Choose DBSCAN Parameters")
    eps = st.slider("EPS (Neighborhood radius)", 0.1, 10.0, 0.5, 0.1)
    min_samples = st.slider("Min Samples", 1, 20, 5)
    with st.spinner("Running DBSCAN..."):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
    clustered = data.copy()
    clustered["Cluster"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    st.write(f"📈 Number of Clusters (excluding noise): {n_clusters}")
    st.write(f"🕳️ Number of Noise Points: {n_noise}")
    st.markdown("### 📉 K-Distance Plot (to help choose EPS)")
    plot_k_distance(X_scaled, min_samples)
    st.markdown("### 📍 2D Cluster Visualization")
    plot_2d(X_scaled, labels)
    if X_scaled.shape[1] >= 3:
        st.markdown("### 🧭 Interactive 3D Cluster Visualization")
        plot_3d_interactive(X_scaled, labels)
    st.markdown("### 📥 Download Clustered Data")
    st.dataframe(clustered)
    st.markdown(download_link(clustered, "clustered_data.csv", "Download CSV"), unsafe_allow_html=True)

def run_kmeans(data: pd.DataFrame):
    st.subheader("📌 K-Means Clustering")
    numeric_data = data.select_dtypes(include='number')
    if numeric_data.shape[1] < 2:
        st.warning("Not enough numeric features for clustering (need at least 2).")
        return
    X_scaled = StandardScaler().fit_transform(numeric_data)
    st.markdown("### 🔧 Choose K-Means Parameters")
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    with st.spinner("Running K-Means..."):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
    clustered = data.copy()
    clustered["Cluster"] = labels
    st.markdown("### 📍 K-Means Cluster Visualization")
    plot_2d(X_scaled, labels)
    st.markdown("### 📥 Download Clustered Data")
    st.dataframe(clustered)
    st.markdown(download_link(clustered, "kmeans_clustered_data.csv", "Download CSV"), unsafe_allow_html=True)

# Main App
st.title("🔍 DBSCAN Clustering Explorer")
st.sidebar.title("Data Upload or Selection")
use_default = st.sidebar.checkbox("Use Iris Dataset", value=True)
if use_default:
    data = load_data(use_default=True)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            data = load_data(file=uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            st.stop()
    else:
        st.warning("Please upload a CSV file or select the default dataset.")
        st.stop()
st.subheader("Raw Data Preview")
st.dataframe(data.head())
run_eda(data)
run_dbscan(data)
run_kmeans(data)
