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
Analyze Chicago crime patterns using Machine Learning.
""")

# Show key metrics
st.markdown("---")
st.subheader("""Problem Statement""")
st.markdown("Chicago is located in the United States Of America. Nowdays, The crime rate is increasing day by day.
The Police Department face lot of challenges when it comes to how to deploy their police resources to prevent the criminal activities.
As an analyst , We help the police department to use their resources efficiently to control crimes by providing data driven insights using unsupervised machine learning models.""")

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
