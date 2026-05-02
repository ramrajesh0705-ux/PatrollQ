# app/Home.py
import streamlit as st
import json
import os

st.set_page_config(
    page_title="PatrolIQ",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚓 PatrolIQ - Smart Safety Analytics Platform")

st.markdown("""
Analyze Chicago crime patterns using Machine Learning & Data Science.
""")

# Show key metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Records Analyzed", "500,000")

with col2:
    st.metric("Crime Types", "33")

with col3:
    st.metric("Districts", "25")

# Load results if available
results_file = "outputs/clustering_results.json"
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    st.subheader("Best Model Results")
    best_k = results["best_kmeans"]["k"]
    best_score = results["best_kmeans"]["silhouette_score"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Optimal Clusters (K)", best_k)
    with col2:
        st.metric("Silhouette Score", f"{best_score:.4f}")
    with col3:
        st.metric("Status", "✓ Ready")
else:
    st.warning("⚠️ Run: python src/train.py first")

st.markdown("---")

st.subheader("📱 Navigation")
st.info("""
Use the **sidebar menu** (←) to navigate:
- **Crime Analysis** - Crime statistics
- **Clustering** - Geographic hotspots
- **Dimensionality** - PCA/t-SNE visualization  
- **MLflow Integration** - Model tracking
""")

st.markdown("---")

st.subheader("💼 Business Use Cases")

tab1, tab2, tab3 = st.tabs(["Police", "City Admin", "Emergency"])

with tab1:
    st.write("""
    - Optimize patrol routes
    - Identify high-risk areas
    - Reduce response time by 60%
    """)

with tab2:
    st.write("""
    - Data-driven planning
    - Budget justification
    - Strategic surveillance placement
    """)

with tab3:
    st.write("""
    - Priority emergency calls
    - Optimize unit deployment
    - Real-time situational awareness
    """)