import streamlit as st
import pandas as pd
import numpy as np
import gower
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🔍 Mixed-Type Clustering with Gower Distance & Agglomerative Clustering")

# --- Upload or Sample Dataset ---
st.sidebar.header("📁 Data Input")
data_source = st.sidebar.radio("Choose dataset:", ["Upload CSV", "Use Sample Titanic Dataset"])

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()
else:
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
    df = df[['sex', 'age', 'fare', 'class', 'embarked', 'alone']].dropna()

st.write("### 🧾 Preview of Dataset")
st.dataframe(df.head())

# --- Clean and Gower Distance ---
df_clean = df.dropna().reset_index(drop=True)
gower_dist = gower.gower_matrix(df_clean)

# --- Determine Optimal K ---
st.sidebar.markdown("## 🔢 Cluster Settings")
max_k = st.sidebar.slider("Max clusters to try (for Silhouette)", 2, 10, 5)
sil_scores = []

with st.spinner("⏳ Calculating silhouette scores..."):
    for k in range(2, max_k + 1):
        model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
        labels = model.fit_predict(gower_dist)
        score = silhouette_score(gower_dist, labels, metric="precomputed")
        sil_scores.append((k, score))

best_k, best_score = max(sil_scores, key=lambda x: x[1])
st.sidebar.markdown(f"🧠 Best K: **{best_k}** (Silhouette: {best_score:.2f})")
k_select = st.sidebar.slider("Select number of clusters", 2, 10, best_k)

# --- Final Model ---
final_model = AgglomerativeClustering(n_clusters=k_select, metric='precomputed', linkage='average')
labels = final_model.fit_predict(gower_dist)
df_clean["cluster"] = labels

# --- 2D Visualization ---
tsne_2d = TSNE(n_components=2, metric="precomputed",init="random", random_state=42).fit_transform(gower_dist)
fig_2d = px.scatter(x=tsne_2d[:, 0], y=tsne_2d[:, 1],
                    color=df_clean["cluster"].astype(str),
                    title="2D t-SNE Clusters", labels={"color": "Cluster"})
st.subheader("🧭 2D Cluster Visualization (t-SNE)")
st.plotly_chart(fig_2d, use_container_width=True)

# --- 3D Visualization ---
tsne_3d = TSNE(n_components=3, metric="precomputed", init="random", random_state=42).fit_transform(gower_dist)
fig_3d = px.scatter_3d(x=tsne_3d[:, 0], y=tsne_3d[:, 1], z=tsne_3d[:, 2],
                       color=df_clean["cluster"].astype(str),
                       title="3D t-SNE Clusters")
st.subheader("🧭 3D Cluster Visualization (t-SNE)")
st.plotly_chart(fig_3d, use_container_width=True)

# --- Silhouette Plot ---
k_vals, scores = zip(*sil_scores)
fig_sil = px.line(x=k_vals, y=scores, markers=True,
                  labels={"x": "Number of Clusters", "y": "Silhouette Score"},
                  title="Silhouette Score vs Number of Clusters")
st.subheader("📈 Silhouette Score Analysis")
st.plotly_chart(fig_sil, use_container_width=True)

# --- Radar Chart (Profile per Cluster) ---
st.subheader("📊 Cluster Profile Radar Chart")
num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
scaled_data = MinMaxScaler().fit_transform(df_clean[num_cols])
df_scaled = pd.DataFrame(scaled_data, columns=num_cols)
df_scaled["cluster"] = df_clean["cluster"]

avg_profiles = df_scaled.groupby("cluster").mean().reset_index()
categories = num_cols

fig_radar = go.Figure()
for i, row in avg_profiles.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=row[categories].values,
        theta=categories,
        fill='toself',
        name=f"Cluster {int(row['cluster'])}"
    ))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
st.plotly_chart(fig_radar, use_container_width=True)

# --- Download Clustered CSV ---
st.subheader("⬇️ Download Labeled Data")
buffer = BytesIO()
df_clean.to_csv(buffer, index=False)
st.download_button(label="Download CSV with Cluster Labels",
                   data=buffer.getvalue(),
                   file_name="clustered_data.csv",
                   mime="text/csv")
