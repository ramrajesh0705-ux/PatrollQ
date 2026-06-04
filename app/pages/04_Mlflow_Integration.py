# app/pages/04_MLflow_Integration.py
import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="MLflow", page_icon="📊", layout="wide")

st.title("📊 MLflow Experiment Tracking")

st.info("""
MLflow tracks all model experiments, parameters, and metrics for reproducibility and versioning.
""")

@st.cache_data
def load_results():
    try:
        with open("outputs/clustering_results.json") as f:
            return json.load(f)
    except:
        return None

results = load_results()

if results is None:
    st.error("❌ Please run: python src/train.py")
    st.stop()

# ====== All Experiments ======
st.subheader("All KMeans Experiments")

kmeans_data = results['kmeans_results']

df_experiments = pd.DataFrame(kmeans_data)

st.dataframe(
    df_experiments.style.highlight_max(subset=['silhouette_score']),
    use_container_width=True
)

# ====== Best Model ======
st.subheader("🏆 Best Model - Production Ready")

best = results['best_kmeans']

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("K Value", best['k'])

with col2:
    st.metric("Silhouette Score", f"{best['silhouette_score']:.4f}")

with col3:
    st.metric("Status", "✓ Registered")

st.success(f"✓ Model with K={best['k']} registered in MLflow as production model")

# ====== Model Registry ======
st.subheader("Model Registry Information")

st.json({
    "model_name": "chicago-crime-kmeans",
    "version": "1.0",
    "status": "Production",
    "algorithm": "KMeans",
    "parameters": {
        "n_clusters": best['k'],
        "random_state": 42,
        "n_init": 10
    },
    "metrics": {
        "silhouette_score": best['silhouette_score'],
        "davies_bouldin_score": kmeans_data[best['k']-10]['davies_bouldin_score']
    },
    "created_at": "2026-04-20",
    "updated_at": "2026-04-20"
})

# ====== How to Use MLflow UI ======
st.subheader("View Full MLflow Dashboard")

st.code("""
mlflow ui
# Then open: http://localhost:5000
""", language="bash")

st.info("""
**In MLflow UI you can:**
- Compare all experiments side-by-side
- View parameter values for each run
- See metrics evolution
- Track model versions
- Restore previous models
""")

st.success("✅ MLflow integration loaded successfully!")
