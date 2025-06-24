import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import trustworthiness



st.set_page_config(layout="wide")
st.title("🔍 K-Means Clustering App (Auto-K, 3D PCA & Profiles)")

# --- 1. Upload or Sample Data ---
st.sidebar.header("📥 Upload CSV or Use Sample")
use_sample = st.sidebar.checkbox("Use Iris Dataset (sample)", value=True)

if use_sample:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    st.success("Loaded sample Iris dataset")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("CSV Loaded")
    else:
        st.warning("Please upload a file or use the sample.")
        st.stop()

st.write("### Data Preview", df.head())

# --- 2. Preprocessing ---
num_df = df.select_dtypes(include="number")
if num_df.shape[1] < 2:
    st.error("Need at least 2 numeric columns for clustering.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(num_df)

# --- 3. Elbow + Silhouette Score + Auto K ---
st.subheader("📈 Elbow Curve & Silhouette Score")

inertia, silhouette, best_k = [], [], 2
K = range(2, 11)

for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    silhouette_val = silhouette_score(X_scaled, labels)
    silhouette.append(silhouette_val)

# Auto-suggest best K
best_k = K[np.argmax(silhouette)]
st.success(f"✅ Suggested optimal number of clusters (K): **{best_k}** (based on Silhouette Score)")

fig, ax1 = plt.subplots(figsize=(10, 4))
color = 'tab:blue'
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia", color=color)
ax1.plot(K, inertia, 'o-', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel("Silhouette Score", color=color)
ax2.plot(K, silhouette, 's--', color=color)
ax2.tick_params(axis='y', labelcolor=color)

st.pyplot(fig)

# --- 4. User Select K ---
st.sidebar.header("🔧 Clustering Settings")
n_clusters = st.sidebar.slider("Choose number of clusters", 2, 10, best_k)

# --- 5. KMeans + PCA ---
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
df["PCA1"] = X_pca_2d[:, 0]
df["PCA2"] = X_pca_2d[:, 1]

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)
df["PCA3"] = X_pca_3d[:, 2]

# --- 6. 2D Plot ---
st.subheader("🧬 2D PCA Cluster Plot")
fig2d, ax2d = plt.subplots()
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", ax=ax2d)
ax2d.set_title("PCA 2D View of Clusters")
st.pyplot(fig2d)

# --- 7. 3D Plot ---
st.subheader("🌐 3D PCA Cluster Plot")
fig3d = px.scatter_3d(df, x='PCA1', y='PCA2', z='PCA3',
                      color='Cluster', symbol='Cluster',
                      title="3D PCA Cluster Visualization")
st.plotly_chart(fig3d, use_container_width=True)

# --- 8. Cluster Centers (Unscaled) ---
st.subheader("📍 Cluster Centers (Original Scale)")
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=num_df.columns)
centers_df["Cluster"] = range(n_clusters)
st.dataframe(centers_df)

# --- 9. Cluster Profiles (Averages + Radar Chart w/ Feature Selection) ---
st.subheader("📊 Cluster Profiles (Feature Means per Cluster)")

# Compute cluster profile means
cluster_profile = df.groupby("Cluster")[num_df.columns].mean().round(2)
st.dataframe(cluster_profile.style.highlight_max(axis=0))

# --- Feature Selection ---
st.subheader("🧮 Feature Selection for Radar Chart")
selected_features = st.multiselect(
    "Select features to compare clusters:",
    options=num_df.columns.tolist(),
    default=num_df.columns.tolist()
)

if len(selected_features) < 2:
    st.warning("Select at least 2 features to generate a radar chart.")
    st.stop()

# --- Normalize selected features for radar comparison ---
normalized_profile = cluster_profile[selected_features].copy()
for col in selected_features:
    min_val = normalized_profile[col].min()
    max_val = normalized_profile[col].max()
    normalized_profile[col] = (normalized_profile[col] - min_val) / (max_val - min_val + 1e-9)

# --- Chart Type Selection ---
st.subheader("🧭 Radar Chart Options")
compare_mode = st.radio("Compare Mode:", ["Compare All Clusters", "Compare One Cluster vs. Rest"])

# --- Initialize Plotly Radar Chart ---
import plotly.graph_objects as go
fig_radar = go.Figure()
theta = selected_features + [selected_features[0]]  # loop back to start

# --- Individual Comparison Mode ---
if compare_mode == "Compare One Cluster vs. Rest":
    chosen_cluster = st.selectbox("Choose cluster to compare:", normalized_profile.index.tolist())

    r_cluster = normalized_profile.loc[chosen_cluster].tolist()
    r_rest = normalized_profile.drop(chosen_cluster).mean().tolist()

    fig_radar.add_trace(go.Scatterpolar(
        r=r_cluster + [r_cluster[0]],
        theta=theta,
        fill='toself',
        name=f"Cluster {chosen_cluster}"
    ))

    fig_radar.add_trace(go.Scatterpolar(
        r=r_rest + [r_rest[0]],
        theta=theta,
        fill='toself',
        name="Rest (Average)"
    ))

# --- All Clusters Comparison ---
else:
    for i, row in normalized_profile.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=row.tolist() + [row.tolist()[0]],
            theta=theta,
            fill='toself',
            name=f"Cluster {i}"
        ))

# --- Final Chart Config ---
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    title="Cluster Profile Radar Chart"
)
st.plotly_chart(fig_radar, use_container_width=True)

# --- 10. T-SNE Visualization ---
st.subheader("🌌 T-SNE Cluster Visualization")

from sklearn.manifold import TSNE

# T-SNE Parameters
tsne_dim = st.radio("Choose T-SNE dimensionality:", [2, 3], horizontal=True)
perplexity = st.slider("T-SNE Perplexity (recommended: 5–50)", 5, 50, 30)
n_iter = st.slider("T-SNE Iterations", 250, 2000, 1000)

# Run T-SNE
tsne = TSNE(n_components=tsne_dim, perplexity=perplexity, n_iter=n_iter, random_state=42)
tsne_results = tsne.fit_transform(num_df)

# --- Compute Trustworthiness ---
trust = trustworthiness(num_df, tsne_results, n_neighbors=5)
st.success(f"🔍 Trustworthiness Score: **{trust:.4f}**")

# Build result DataFrame
tsne_df = pd.DataFrame(tsne_results, columns=[f'TSNE-{i+1}' for i in range(tsne_dim)])
tsne_df['Cluster'] = df['Cluster'].astype(str)

# Plot
if tsne_dim == 2:
    fig_tsne = px.scatter(
        tsne_df, x='TSNE-1', y='TSNE-2',
        color='Cluster',
        title='T-SNE 2D Visualization of Clusters',
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.7, width=800, height=500
    )
else:
    fig_tsne = px.scatter_3d(
        tsne_df, x='TSNE-1', y='TSNE-2', z='TSNE-3',
        color='Cluster',
        title='T-SNE 3D Visualization of Clusters',
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.7, width=900, height=600
    )

st.plotly_chart(fig_tsne, use_container_width=True)


# --- 11. Download Results ---
st.sidebar.header("⬇️ Download")
csv_out = df.to_csv(index=False)
st.sidebar.download_button("Download Clustered Data", csv_out, file_name="clustered_data.csv", mime="text/csv")
