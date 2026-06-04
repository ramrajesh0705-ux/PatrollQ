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

# District name mapping for cluster naming
district_name_mapping = {
    1.0: 'Central/Loop',
    2.0: 'Wentworth (South Side)',
    3.0: 'Grand Crossing (South Side)',
    4.0: 'South Chicago (South Side)',
    5.0: 'Calumet (Far South Side)',
    6.0: 'Gresham (South Side)',
    7.0: 'Englewood (South Side)',
    8.0: 'Chicago Lawn (Southwest Side)',
    9.0: 'Deering (Southwest Side)',
    10.0: 'Ogden (West Side)',
    11.0: 'Harrison (West Side)',
    12.0: 'Near West (West Side)',
    14.0: 'Shakespeare (Near North/West)',
    15.0: 'Austin (West Side)',
    16.0: 'Jefferson Park (Northwest Side)',
    17.0: 'Albany Park (North Side)',
    18.0: 'Near North (Near North Side)',
    19.0: 'Town Hall (North Side)',
    20.0: 'Lincoln (North Side)',
    22.0: 'Morgan Park (Far South Side)',
    24.0: 'Rogers Park (Far North Side)',
    25.0: 'Grand Central (Northwest Side)',
    31.0: 'Chicago Police Academy (Training/Other)'
}

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
        
        # Calculate cluster names based on most frequent district
        cluster_names = {}
        cluster_districts = {}
        
        for cluster_id in range(best_k):
            cluster_data = valid_data[valid_data['Cluster'] == cluster_id]
            # Get most frequent district in this cluster
            most_frequent_district = cluster_data['District'].mode()
            if len(most_frequent_district) > 0:
                district_val = most_frequent_district.iloc[0]
                cluster_districts[cluster_id] = district_val
                cluster_names[cluster_id] = district_name_mapping.get(district_val, f"Cluster {cluster_id}")
            else:
                cluster_names[cluster_id] = f"Cluster {cluster_id}"
        
        # Create cluster summary DataFrame
        cluster_summary = []
        for cluster_id in range(best_k):
            cluster_data = valid_data[valid_data['Cluster'] == cluster_id]
            crime_count = len(cluster_data)
            
            # Most frequent crime type
            most_frequent_crime = cluster_data['Primary Type'].mode()
            crime_type = most_frequent_crime.iloc[0] if len(most_frequent_crime) > 0 else "Unknown"
            
            # Peak hour
            peak_hour = cluster_data['Hour'].mode()
            peak = int(peak_hour.iloc[0]) if len(peak_hour) > 0 else 0
            
            # Calculate percentage (fixed the rounding issue)
            percentage = round((crime_count / len(valid_data) * 100), 1)
            
            cluster_summary.append({
                'Cluster ID': cluster_id,
                'Cluster Name': cluster_names[cluster_id],
                'Crime Count': crime_count,
                'Most Frequent Primary Type': crime_type,
                'Peak Hour': peak,
                'Percentage': percentage
            })
        
        # Convert to DataFrame and sort by Crime Count
        cluster_summary_df = pd.DataFrame(cluster_summary)
        cluster_summary_df = cluster_summary_df.sort_values('Crime Count', ascending=False)
        
        # Map visualization with cluster names in hover
        sample_cluster = valid_data.sample(min(5000, len(valid_data)), random_state=42)
        
        # Add cluster names to the sample data
        sample_cluster['Cluster Name'] = sample_cluster['Cluster'].map(cluster_names)
        
        fig = px.scatter_mapbox(
            sample_cluster,
            lat='Latitude',
            lon='Longitude',
            color='Cluster',
            hover_name='Cluster Name',
            hover_data={'Cluster': True, 'Primary Type': True, 'Hour': True},
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
        
        # Most frequent crime types per cluster with names
        st.subheader("🔍 Most Frequent Crime Types by Cluster")
        
        for _, row in cluster_summary_df.iterrows():
            cluster_id = row['Cluster ID']
            cluster_name = row['Cluster Name']
            cluster_data = valid_data[valid_data['Cluster'] == cluster_id]
            top_crimes = cluster_data['Primary Type'].value_counts().head(5)
            
            with st.expander(f"📍 {cluster_name} - {row['Crime Count']:,} crimes ({row['Percentage']:.1f}%)"):
                crime_df = pd.DataFrame({
                    'Crime Type': top_crimes.index,
                    'Count': top_crimes.values,
                    'Percentage': (top_crimes.values / len(cluster_data) * 100).round(1)
                })
                st.dataframe(crime_df, use_container_width=True, hide_index=True)
        
        # Peak hour analysis with cluster names
        st.subheader("⏰ Peak Hours by Cluster")
        
        peak_hours = []
        for _, row in cluster_summary_df.iterrows():
            peak_hours.append({
                'Cluster Name': row['Cluster Name'],
                'Peak Hour': f"{int(row['Peak Hour'])}:00",
                'Crime Count': row['Crime Count'],
                'Most Frequent Crime': row['Most Frequent Primary Type']
            })
        
        peak_df = pd.DataFrame(peak_hours)
        st.dataframe(peak_df, use_container_width=True, hide_index=True)
        
        # Save to session state
        st.session_state['clustering_results'] = valid_data
        st.session_state['kmeans_model'] = kmeans
        st.session_state['best_k'] = best_k
        st.session_state['cluster_summary'] = cluster_summary_df
        
    else:
        st.error("No valid coordinate data available for clustering")

# Optional: Show summary if clustering results exist
if 'clustering_results' in st.session_state:
    st.subheader("📈 Cluster Analysis Summary")
    cluster_data = st.session_state['clustering_results']
    cluster_summary_df = st.session_state['cluster_summary']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Clusters", st.session_state['best_k'])
    
    with col2:
        st.metric("Total Records Clustered", f"{len(cluster_data):,}")
    
    with col3:
        # Calculate average crimes per cluster
        avg_crimes = len(cluster_data) / st.session_state['best_k']
        st.metric("Avg Crimes per Cluster", f"{avg_crimes:,.0f}")
    
    with col4:
        # Largest cluster
        largest_cluster = cluster_summary_df.iloc[0]
        st.metric("Largest Cluster", largest_cluster['Cluster Name'])

st.success("✅ Clustering analysis loaded successfully!")
