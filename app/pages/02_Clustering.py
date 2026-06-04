# app/pages/02_Clustering.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np

st.set_page_config(page_title="Clustering", page_icon="🗺️", layout="wide")

st.title("🗺️ Crime Clustering Analysis")

@st.cache_data
def load_data():
    """Load and preprocess the crime data"""
    data_path = "data/processed/crime_cleaned.csv"
    if not os.path.exists(data_path):
        return None
    
    df = pd.read_csv(data_path)
    
    # Convert 'Date' to datetime, allowing inference. Errors will turn them to NaT
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
    
    # Ensure 'month' is not NaN before applying get_season
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

def perform_clustering(filtered, optimal_k=10):
    """Perform K-means clustering on crime locations"""
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Apply Min-Max scaling to 'Latitude' and 'Longitude'
    filtered[['Latitude', 'Longitude']] = scaler.fit_transform(filtered[['Latitude', 'Longitude']])
    
    # Run K-Means clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    filtered['cluster'] = kmeans.fit_predict(filtered[['Latitude', 'Longitude']])
    
    return filtered, kmeans, scaler


def plot_clusters(filtered, kmeans):
    """Plot the clustering results"""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Longitude', y='Latitude', hue='cluster', data=filtered, palette='viridis', s=20, alpha=0.6)
    
    # Plot the cluster centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 1], centroids[:, 0], marker='X', s=200, color='red', edgecolor='black', label='Centroids')
    
    plt.title('K-Means Clustering with Centroids')
    plt.xlabel('Scaled Longitude')
    plt.ylabel('Scaled Latitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Cluster')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def analyze_clusters(filtered):
    """Analyze cluster characteristics"""
    # Crime count by cluster
    crime_by_cluster = filtered['cluster'].value_counts().sort_index()
    print("Crime count by cluster:")
    print(crime_by_cluster)
    
    # Most frequent crime type by cluster
    most_frequent_crime_by_cluster = filtered.groupby('cluster')['Primary Type'].agg(lambda x: x.mode()[0])
    print("\nMost frequent crime type by cluster:")
    print(most_frequent_crime_by_cluster)
    
    # Peak time analysis by cluster
    peak_time_by_cluster_crime = {}
    
    for cluster_id, most_frequent_type in most_frequent_crime_by_cluster.items():
        # Filter data for the current cluster and its most frequent crime type
        cluster_crime_df = filtered[
            (filtered['cluster'] == cluster_id) &
            (filtered['Primary Type'] == most_frequent_type)
        ]
        
        # Count occurrences of 'Hour'
        hour_counts = cluster_crime_df['Hour'].value_counts()
        
        if not hour_counts.empty:
            peak_hour = hour_counts.idxmax()
            peak_time_by_cluster_crime[cluster_id] = {
                'Most Frequent Primary Type': most_frequent_type,
                'Peak Hour': peak_hour
            }
        else:
            peak_time_by_cluster_crime[cluster_id] = {
                'Most Frequent Primary Type': most_frequent_type,
                'Peak Hour': 'No data available'
            }
    
    # Convert to DataFrame for better display
    peak_time_df = pd.DataFrame.from_dict(peak_time_by_cluster_crime, orient='index')
    peak_time_df.index.name = 'Cluster'
    print("\nPeak time analysis:")
    print(peak_time_df)
    
    # Most frequent district by cluster
    most_frequent_district_by_cluster = filtered.groupby('cluster')['District'].agg(lambda x: x.mode()[0])
    print("\nMost frequent district by cluster:")
    print(most_frequent_district_by_cluster)
    
    return crime_by_cluster, peak_time_df, most_frequent_district_by_cluster


def create_cluster_summary(filtered):
    """Create final cluster summary with location names"""
    crime_by_cluster, peak_time_df, district_df = analyze_clusters(filtered)
    
    final_cluster_summary = pd.DataFrame(crime_by_cluster).rename(columns={'count': 'Crime Count'})
    final_cluster_summary = final_cluster_summary.merge(peak_time_df, left_index=True, right_index=True)
    final_cluster_summary = final_cluster_summary.merge(district_df.rename('Most Frequent District'), left_index=True, right_index=True)
    
    # District name mapping
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
    
    # Map the 'Most Frequent District' to these names
    final_cluster_summary['Location Name'] = final_cluster_summary['Most Frequent District'].map(district_name_mapping)
    
    # Create a consolidated 'Cluster Name'
    final_cluster_summary['Cluster Name'] = final_cluster_summary['Location Name']
    
    # Display final summary
    print("\nFinal Cluster Summary:")
    print(final_cluster_summary[['Cluster Name', 'Crime Count', 'Most Frequent Primary Type', 'Peak Hour']])
    
    return final_cluster_summary


def plot_crime_by_month(df):
    """Plot crime distribution by month"""
    plt.figure(figsize=(10, 6))
    monthly_crimes = df.groupby('Month').size()
    plt.plot(monthly_crimes.index, monthly_crimes.values, marker='o', linewidth=2, markersize=8)
    plt.title('Crime Incidents by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Incidents')
    plt.xticks(range(1, 13), calendar.month_name[1:13], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_crime_by_day(df):
    """Plot crime distribution by day of week"""
    plt.figure(figsize=(10, 6))
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_crimes = df.groupby('DayOfWeek').size().reindex(days_order)
    
    plt.bar(daily_crimes.index, daily_crimes.values, color='skyblue', edgecolor='navy')
    plt.title('Crime Incidents by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Incidents')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the analysis"""
    # Change to working directory
    os.chdir('/content/drive/MyDrive/PatrolQ')
    
    # Load data
    print("Loading data...")
    filtered = load_data()
    
    if filtered is None:
        print("Error: Could not load data. Please check the file path.")
        return
    
    print(f"Data loaded successfully. Shape: {filtered.shape}")
    print(f"Columns: {filtered.columns.tolist()}")
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(filtered.head())
    
    # Display unique values in categorical columns
    print("\nUnique Wards:", filtered['Ward'].unique())
    print("Unique Districts:", filtered['District'].unique())
    print("Unique Community Areas:", filtered['Community Area'].unique())
    
    # Plot crime by month
    print("\nGenerating crime by month plot...")
    plot_crime_by_month(filtered)
    
    # Plot crime by day of week
    print("\nGenerating crime by day of week plot...")
    plot_crime_by_day(filtered)
    
    # Perform clustering
    print("\nPerforming clustering...")
    filtered, kmeans, scaler = perform_clustering(filtered, optimal_k=10)
    
    
    # Plot clusters
    print("\nPlotting clusters...")
    plot_clusters(filtered, kmeans)
    
    # Analyze clusters
    print("\nAnalyzing clusters...")
    final_summary = create_cluster_summary(filtered)
    
 
    return filtered, final_summary


if __name__ == "__main__":
    main()
