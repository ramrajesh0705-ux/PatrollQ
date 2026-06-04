# app/pages/02_Clustering.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Clustering", page_icon="🗺️", layout="wide")

st.title("🗺️ Crime Clustering Analysis")

@st.cache_data
def load_data():
    """Load and preprocess the crime data"""
    data_path = "data/processed/crime_cleaned.csv"
    if not os.path.exists(data_path):
        return None
    
    df = pd.read_csv(data_path)
    
    # Convert coordinates to numeric
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Arrest"] = df["Arrest"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
    df["Domestic"] = df["Domestic"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
    df["Primary Type"] = df["Primary Type"].astype("category")
    
    def get_season(month: int) -> str:
        if month in [12, 1, 2]:
            return "Winter"
        if month in [3, 4, 5]:
            return "Spring"
        if month in [6, 7, 8]:
            return "Summer"
        return "Fall"
    
    df["season"] = df["Month"].apply(lambda x: get_season(int(x)) if pd.notna(x) else None)
    
    return df

df = load_data()

@st.cache_data
def load_results():
    try:
        with open("outputs/clustering_results.json", 'r') as f:
            return json.load(f)
    except:
        return None

results = load_results()

if results is None:
    st.error("❌ Please run: python src/train.py")
    st.stop()

if df is None:
    st.error("❌ Data file not found")
    st.stop()

# ====== KMeans Performance ======
st.subheader("K-Means Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    kmeans_data = results['kmeans_results']
    k_values = [r['k'] for r in kmeans_data]
    silhouette_scores = [r['silhouette_score'] for r in kmeans_data]
    
    fig = px.line(
        x=k_values,
        y=silhouette_scores,
        markers=True,
        title="Silhouette Score by Number of Clusters",
        labels={"x": "Number of Clusters (K)", "y": "Silhouette Score"},
        color_discrete_sequence=['#1f77b4']
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                   annotation_text="Good Threshold (0.5)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    davies_bouldin_scores = [r['davies_bouldin_score'] for r in kmeans_data]
    
    fig = px.line(
        x=k_values,
        y=davies_bouldin_scores,
        markers=True,
        title="Davies-Bouldin Index by Number of Clusters",
        labels={"x": "Number of Clusters (K)", "y": "Davies-Bouldin Index"},
        color_discrete_sequence=['#ff7f0e']
    )
    st.plotly_chart(fig, use_container_width=True)

# ====== Best Model Info ======
st.subheader("🏆 Best Model")

best_k = results['best_kmeans']['k']
best_score = results['best_kmeans']['silhouette_score']

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Optimal K", best_k)

with col2:
    st.metric("Silhouette Score", f"{best_score:.4f}")

with col3:
    st.metric("Status", "✓ Production Ready")

st.success(f"**K={best_k}** selected as best model with silhouette score of **{best_score:.4f}**")

# ====== Geographic Visualization ======
st.subheader("📍 Geographic Crime Distribution with Clusters")

# Perform clustering using best_k
with st.spinner(f"Performing clustering with K={best_k}..."):
    # Filter out rows with missing coordinates
    valid_data = df.dropna(subset=['Latitude', 'Longitude']).copy()
    
    if len(valid_data) > 0:
        # Perform clustering
        scaler = MinMaxScaler()
        valid_data[['Latitude_scaled', 'Longitude_scaled']] = scaler.fit_transform(
            valid_data[['Latitude', 'Longitude']]
        )
        
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        valid_data['Cluster'] = kmeans.fit_predict(valid_data[['Latitude_scaled', 'Longitude_scaled']])
        
        # Display cluster statistics
        st.success(f"✅ Clustering completed with {best_k} clusters")
        
        # Cluster sizes
        cluster_sizes = valid_data['Cluster'].value_counts().sort_index()
        
        # Create metrics for each cluster
        cols = st.columns(min(4, best_k))
        for i, (cluster_id, size) in enumerate(cluster_sizes.items()):
            with cols[i % len(cols)]:
                percentage = (size / len(valid_data)) * 100
                st.metric(
                    f"Cluster {cluster_id}",
                    f"{size:,} crimes",
                    f"{percentage:.1f}% of total"
                )
        
        # Map visualization
        sample_cluster = valid_data.sample(min(5000, len(valid_data)), random_state=42)
        
        fig = px.scatter_mapbox(
            sample_cluster,
            lat='Latitude',
            lon='Longitude',
            color='Cluster',
            title=f"Crime Clusters (K={best_k}) - Sample of {len(sample_cluster):,} points",
            zoom=10,
            height=600,
            opacity=0.7,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0, "t":40, "l":0, "b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Most frequent crime types per cluster
        st.subheader("🔍 Most Frequent Crime Types by Cluster")
        
        for cluster_id in range(best_k):
            if cluster_id in cluster_sizes.index:
                cluster_data = valid_data[valid_data['Cluster'] == cluster_id]
                top_crimes = cluster_data['Primary Type'].value_counts().head(3)
                
                with st.expander(f"📍 Cluster {cluster_id} - {cluster_sizes[cluster_id]:,} crimes ({cluster_sizes[cluster_id]/len(valid_data)*100:.1f}%)"):
                    crime_df = pd.DataFrame({
                        'Crime Type': top_crimes.index,
                        'Count': top_crimes.values,
                        'Percentage': (top_crimes.values / len(cluster_data) * 100).round(1)
                    })
                    st.dataframe(crime_df, use_container_width=True, hide_index=True)
        
        # Peak hour analysis
        st.subheader("⏰ Peak Hours by Cluster")
        
        peak_hours = []
        for cluster_id in range(best_k):
            if cluster_id in cluster_sizes.index:
                cluster_data = valid_data[valid_data['Cluster'] == cluster_id]
                peak_hour = cluster_data['Hour'].mode()
                if len(peak_hour) > 0:
                    peak_hours.append({
                        'Cluster': cluster_id,
                        'Peak Hour': f"{int(peak_hour.iloc[0])}:00",
                        'Crime Count': len(cluster_data)
                    })
        
        if peak_hours:
            peak_df = pd.DataFrame(peak_hours)
            st.dataframe(peak_df, use_container_width=True, hide_index=True)
        
        # Save to session state
        st.session_state['clustering_results'] = valid_data
        st.session_state['kmeans_model'] = kmeans
        st.session_state['best_k'] = best_k
        
    else:
        st.error("No valid coordinate data available for clustering")

# Optional: Show summary if clustering results exist
if 'clustering_results' in st.session_state:
    st.subheader("📈 Cluster Analysis Summary")
    cluster_data = st.session_state['clustering_results']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Clusters", st.session_state['best_k'])
    
    with col2:
        st.metric("Total Records Clustered", f"{len(cluster_data):,}")
    
    with col3:
        # Calculate average crimes per cluster
        avg_crimes = len(cluster_data) / st.session_state['best_k']
        st.metric("Avg Crimes per Cluster", f"{avg_crimes:,.0f}")

st.success("✅ Clustering analysis loaded successfully!")
