import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="🔎", layout="wide")

st.title("🔎 Chicago Crime Exploratory Analysis")
st.markdown(
    "This page converts the exploratory notebook into an interactive Streamlit dashboard. "
    "Use the controls to filter the dataset and explore crime patterns by type, geography, time, arrest status, and domestic incidents."
)

@st.cache_data
def load_data():
    """Load and preprocess data with optimized dtypes"""
    data_path = "data/processed/crime_cleaned.csv"
    if not os.path.exists(data_path):
        return None
    
    # Load only necessary columns initially
    required_cols = ["Date", "Primary Type", "Arrest", "Domestic", "District", 
                     "Community Area", "Latitude", "Longitude", "Year"]
    
    # Use low_memory and specify dtypes for efficiency
    df = pd.read_csv(data_path, low_memory=False)
    
    # Convert to optimized dtypes
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Arrest"] = df["Arrest"].astype(str).str.upper().map({"TRUE": True, "FALSE": False}).astype('boolean')
    df["Domestic"] = df["Domestic"].astype(str).str.upper().map({"TRUE": True, "FALSE": False}).astype('boolean')
    df["Primary Type"] = df["Primary Type"].astype("category")
    df["District"] = pd.to_numeric(df["District"], errors="coerce").astype('Int64')
    df["Community Area"] = pd.to_numeric(df["Community Area"], errors="coerce").astype('Int64')
    
    # Add derived columns efficiently
    if "Date" in df.columns:
        df["month"] = df["Date"].dt.month
        df["hour"] = df["Date"].dt.hour
        df["day_name"] = df["Date"].dt.day_name()
        df["Year"] = df["Date"].dt.year
    
    # Define season function
    def get_season(month: int) -> str:
        if month in [12, 1, 2]:
            return "Winter"
        if month in [3, 4, 5]:
            return "Spring"
        if month in [6, 7, 8]:
            return "Summer"
        return "Fall"
    
    df["season"] = df["month"].apply(get_season).astype("category")
    
    return df

@st.cache_data
def precompute_aggregations(df):
    """Precompute all aggregations to avoid repeated calculations"""
    aggregations = {}
    
    # Crime distribution
    aggregations['crime_counts'] = df["Primary Type"].value_counts()
    
    # Geographic aggregations
    aggregations['district_counts'] = df["District"].value_counts().sort_index()
    aggregations['community_counts'] = df["Community Area"].value_counts().sort_index()
    
    # Temporal aggregations
    aggregations['crimes_per_year'] = df["Year"].value_counts().sort_index()
    aggregations['crimes_by_month'] = df["month"].value_counts().reindex(range(1, 13), fill_value=0)
    aggregations['crimes_by_day'] = df["day_name"].value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fill_value=0
    )
    aggregations['crimes_by_hour'] = df["hour"].value_counts().sort_index()
    
    # Seasonal
    aggregations['season_counts'] = df["season"].value_counts().reindex(["Winter", "Spring", "Summer", "Fall"], fill_value=0)
    
    # Arrest and domestic
    aggregations['arrest_counts'] = df["Arrest"].value_counts()
    aggregations['domestic_counts'] = df["Domestic"].value_counts()
    aggregations['arrest_domestic_cross'] = pd.crosstab(df["Domestic"], df["Arrest"], normalize="index") * 100
    aggregations['domestic_by_hour'] = df[df["Domestic"] == True].groupby("hour").size().reindex(range(24), fill_value=0)
    
    # Year-month heatmap
    aggregations['year_month_pivot'] = df.groupby(["Year", "month"]).size().reset_index(name="count").pivot(
        index="Year", columns="month", values="count"
    ).fillna(0)
    
    # Hourly patterns by year (limit years for performance)
    top_years = df["Year"].value_counts().head(5).index
    aggregations['hour_year'] = df[df["Year"].isin(top_years)].groupby(["Year", "hour"]).size().reset_index(name="count")
    
    return aggregations

# Load data
df = load_data()
if df is None:
    st.error("❌ Could not find crime_cleaned.csv in the data/processed/ directory.")
    st.stop()

# Precompute aggregations
with st.spinner("Loading and processing data..."):
    aggregations = precompute_aggregations(df)

st.markdown(f"**Dataset loaded:** {len(df):,} records")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    
    # Use session state for sample size
    if 'sample_size' not in st.session_state:
        st.session_state.sample_size = min(50000, len(df))
    
    sample_size = st.slider(
        "Sample size for maps", 
        10000, 
        min(50000, len(df)), 
        st.session_state.sample_size, 
        step=5000,
        help="Larger samples show more detail but take longer to render"
    )
    st.session_state.sample_size = sample_size
    
    year_options = sorted(df["Year"].dropna().unique())
    selected_year = st.selectbox("Year", ["All"] + list(year_options), index=0)
    selected_day = st.selectbox("Day of week", ["All"] + ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=0)
    selected_season = st.selectbox("Season", ["All", "Winter", "Spring", "Summer", "Fall"], index=0)

# Apply filters efficiently using boolean indexing
@st.cache_data
def apply_filters(_df, year, day, season):
    """Apply filters with caching"""
    mask = pd.Series([True] * len(_df), index=_df.index)
    if year != "All":
        mask &= (_df["Year"] == int(year))
    if day != "All":
        mask &= (_df["day_name"] == day)
    if season != "All":
        mask &= (_df["season"] == season)
    return _df[mask]

filtered = apply_filters(df, selected_year, selected_day, selected_season)

if len(filtered) == 0:
    st.warning("No records match the selected filters.")
    st.stop()

st.markdown("---")

# 1. Crime Distribution - Use precomputed data with filtering
st.header("1. Crime Distribution")

# Filter crime counts based on filtered data
filtered_crime_counts = filtered["Primary Type"].value_counts()

st.subheader("Crime frequency by type")
# FIX 1: Convert to DataFrame for Plotly Express
crime_counts_df = filtered_crime_counts.reset_index()
crime_counts_df.columns = ["crime_type", "count"]
fig_crime = px.bar(
    crime_counts_df,
    x="count",
    y="crime_type",
    orientation="h",
    title="Crime Count by Primary Type",
    labels={"count": "Count", "crime_type": "Primary Type"},
    color="count",
    color_continuous_scale="Blues"
)
fig_crime.update_layout(yaxis=dict(autorange="reversed"), height=600)
st.plotly_chart(fig_crime, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 10 Crime Types")
    top10 = filtered_crime_counts.head(10).reset_index()
    top10.columns = ["crime_type", "count"]
    fig_top10 = px.bar(
        top10,
        x="count",
        y="crime_type",
        orientation="h",
        labels={"count": "Count", "crime_type": "Crime Type"},
        title="Top 10 Crime Types"
    )
    fig_top10.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_top10, use_container_width=True)
    
with col2:
    st.subheader("Least Frequent Crime Types")
    rare10 = filtered_crime_counts[filtered_crime_counts > 0].tail(10).reset_index()
    if len(rare10) > 0:
        rare10.columns = ["crime_type", "count"]
        fig_rare = px.bar(
            rare10,
            x="count",
            y="crime_type",
            orientation="h",
            labels={"count": "Count", "crime_type": "Crime Type"},
            title="Rare Crime Types"
        )
        fig_rare.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_rare, use_container_width=True)

st.markdown("---")

# 2. Geographic Patterns
st.header("2. Geographic Crime Patterns")

st.subheader("Crime distribution by police district and community area")

# Use filtered data for district/community counts
district_counts = filtered["District"].value_counts().sort_index()
community_counts = filtered["Community Area"].value_counts().sort_index()

col1, col2 = st.columns(2)
with col1:
    if len(district_counts) > 0:
        # FIX 2: Convert to DataFrame
        district_df = district_counts.reset_index()
        district_df.columns = ["district", "count"]
        district_df["district"] = district_df["district"].astype(str)
        fig_dist = px.bar(
            district_df,
            x="district",
            y="count",
            labels={"district": "District", "count": "Number of Crimes"},
            title="Crime by Police District",
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
with col2:
    if len(community_counts) > 0:
        # Limit to top 20 communities for readability
        top_communities = community_counts.head(20).reset_index()
        top_communities.columns = ["community_area", "count"]
        top_communities["community_area"] = top_communities["community_area"].astype(str)
        fig_comm = px.bar(
            top_communities,
            x="community_area",
            y="count",
            labels={"community_area": "Community Area", "count": "Number of Crimes"},
            title="Crime by Community Area (Top 20)",
        )
        st.plotly_chart(fig_comm, use_container_width=True)

# Maps - Use sampling and caching
@st.cache_data
def get_map_sample(_filtered, sample_size, random_state=42):
    """Cache map samples for better performance"""
    map_data = _filtered.dropna(subset=["Latitude", "Longitude"]).copy()
    if len(map_data) > sample_size:
        return map_data.sample(n=sample_size, random_state=random_state)
    return map_data

st.subheader("Spatial density and location clustering")
map_sample = get_map_sample(filtered, sample_size)

if len(map_sample) > 0:
    # Density map
    fig_map = px.density_mapbox(
        map_sample,
        lat="Latitude",
        lon="Longitude",
        radius=8,  # Reduced radius for better performance
        zoom=10,
        height=500,  # Reduced height
        title="Crime Density Map",
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Cluster map - Use aggregated data
    with st.spinner("Generating cluster map..."):
        # Bin coordinates for clustering
        map_sample["lat_bin"] = map_sample["Latitude"].round(3)  # Less precision for fewer bins
        map_sample["lon_bin"] = map_sample["Longitude"].round(3)
        grid_counts = map_sample.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="crime_count")
        
        # Limit points for performance
        if len(grid_counts) > 2000:
            grid_counts = grid_counts.nlargest(2000, "crime_count")
        
        fig_scatter = px.scatter_mapbox(
            grid_counts,
            lat="lat_bin",
            lon="lon_bin",
            size="crime_count",
            size_max=12,
            zoom=10,
            height=500,
            title="Crime Location Clusters",
        )
        fig_scatter.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.warning("No latitude/longitude data available for the selected filters.")

st.markdown("---")

# 3. Temporal Trends
st.header("3. Temporal Crime Trends")

# Use filtered aggregations
crimes_per_year = filtered["Year"].value_counts().sort_index()
if len(crimes_per_year) > 0:
    # FIX 3: Convert to DataFrame
    year_df = crimes_per_year.reset_index()
    year_df.columns = ["year", "count"]
    fig_year = px.line(
        year_df,
        x="year",
        y="count",
        markers=True,
        title="Total Crimes Per Year",
        labels={"year": "Year", "count": "Crime Count"}
    )
    st.plotly_chart(fig_year, use_container_width=True)

# FIX 4: Convert month data to DataFrame
crimes_by_month = filtered["month"].value_counts().reindex(range(1, 13), fill_value=0)
month_df = crimes_by_month.reset_index()
month_df.columns = ["month", "count"]
fig_month = px.bar(
    month_df,
    x="month",
    y="count",
    labels={"month": "Month", "count": "Number of Crimes"},
    title="Crime Volume by Month"
)
st.plotly_chart(fig_month, use_container_width=True)

# Year-month heatmap (this one is fine - uses go.Heatmap directly)
year_month_data = filtered.groupby(["Year", "month"]).size().reset_index(name="count")
if not year_month_data.empty:
    heat = year_month_data.pivot(index="Year", columns="month", values="count").fillna(0)
    fig_heat = go.Figure(data=go.Heatmap(
        z=heat.values,
        x=heat.columns,
        y=heat.index,
        colorscale="Viridis",
        colorbar=dict(title="Count")
    ))
    fig_heat.update_layout(
        title="Crime Trends by Year and Month",
        xaxis_title="Month",
        yaxis_title="Year",
        height=500
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# Day of week and hour patterns
# FIX 5: Convert day of week data to DataFrame
crime_by_day = filtered["day_name"].value_counts().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fill_value=0
)
day_df = crime_by_day.reset_index()
day_df.columns = ["day_of_week", "count"]
fig_day = px.bar(
    day_df,
    x="day_of_week",
    y="count",
    labels={"day_of_week": "Day of Week", "count": "Number of Crimes"},
    title="Crime Counts by Day of Week"
)
st.plotly_chart(fig_day, use_container_width=True)

# FIX 6: This was the main error - convert hour data to DataFrame
crime_by_hour = filtered["hour"].value_counts().sort_index()
hour_df = crime_by_hour.reset_index()
hour_df.columns = ["hour", "count"]
fig_hour = px.line(
    hour_df,
    x="hour",
    y="count",
    markers=True,
    title="Crime Patterns by Hour",
    labels={"hour": "Hour of Day", "count": "Crime Count"}
)
fig_hour.update_xaxes(tickmode="linear")
st.plotly_chart(fig_hour, use_container_width=True)

# Hourly patterns by year (limit to top years)
top_years = filtered["Year"].value_counts().head(5).index
if len(top_years) > 1:
    crimes_hour_year = filtered[filtered["Year"].isin(top_years)].groupby(["Year", "hour"]).size().reset_index(name="count")
    if not crimes_hour_year.empty:
        fig_hour_year = px.line(
            crimes_hour_year,
            x="hour",
            y="count",
            color="Year",
            markers=True,
            title="Hourly Crimes by Year (Top 5 Years)",
            labels={"hour": "Hour", "count": "Crime Count", "Year": "Year"}
        )
        fig_hour_year.update_xaxes(tickmode="linear")
        st.plotly_chart(fig_hour_year, use_container_width=True)

st.markdown("---")

# 4. Seasonal Analysis
st.header("4. Seasonal Crime Patterns")
# FIX 7: Convert season data to DataFrame
season_counts = filtered["season"].value_counts().reindex(["Winter", "Spring", "Summer", "Fall"], fill_value=0)
season_df = season_counts.reset_index()
season_df.columns = ["season", "count"]
fig_season = px.bar(
    season_df,
    x="season",
    y="count",
    labels={"season": "Season", "count": "Number of Crimes"},
    title="Crime Count by Season"
)
st.plotly_chart(fig_season, use_container_width=True)

st.markdown("---")

# 5. Arrest and Domestic Analysis
st.header("5. Arrest and Domestic Incident Analysis")

arrest_counts = filtered["Arrest"].value_counts()
arrest_pct = (arrest_counts / arrest_counts.sum() * 100).round(2)
arrest_labels = {True: "Arrested", False: "Not Arrested"}
arrest_pct.index = [arrest_labels.get(x, str(x)) for x in arrest_pct.index]

domestic_counts = filtered["Domestic"].value_counts()
domestic_pct = (domestic_counts / domestic_counts.sum() * 100).round(2)
domestic_labels = {True: "Domestic", False: "Non-Domestic"}
domestic_pct.index = [domestic_labels.get(x, str(x)) for x in domestic_pct.index]

col1, col2 = st.columns(2)
with col1:
    if len(arrest_pct) > 0:
        # FIX 8: Convert to DataFrame for pie chart (optional but consistent)
        arrest_df = arrest_pct.reset_index()
        arrest_df.columns = ["status", "percentage"]
        fig_arrest = px.pie(
            arrest_df,
            names="status",
            values="percentage",
            title=f"Arrest Rate: {arrest_pct.get('Arrested', 0):.1f}%",
            hole=0.4,
        )
        st.plotly_chart(fig_arrest, use_container_width=True)
        
with col2:
    if len(domestic_pct) > 0:
        # FIX 9: Convert to DataFrame for pie chart
        domestic_df = domestic_pct.reset_index()
        domestic_df.columns = ["status", "percentage"]
        fig_domestic = px.pie(
            domestic_df,
            names="status",
            values="percentage",
            title=f"Domestic Incidents: {domestic_pct.get('Domestic', 0):.1f}%",
            hole=0.4,
        )
        st.plotly_chart(fig_domestic, use_container_width=True)

# Arrest rate by domestic incident (this one already uses DataFrame - fine)
arrest_domestic = pd.crosstab(filtered["Domestic"], filtered["Arrest"], normalize="index") * 100
if not arrest_domestic.empty:
    arrest_domestic = arrest_domestic.rename(index={True: "Domestic", False: "Non-Domestic"})
    arrest_domestic.columns = ["Not Arrested" if c is False else "Arrested" for c in arrest_domestic.columns]
    arrest_domestic = arrest_domestic.reset_index().melt(id_vars="Domestic", var_name="Arrest Status", value_name="Percent")
    fig_ad = px.bar(
        arrest_domestic,
        x="Domestic",
        y="Percent",
        color="Arrest Status",
        title="Arrest Percent by Domestic Incident",
        barmode="stack",
    )
    st.plotly_chart(fig_ad, use_container_width=True)

# Domestic crimes by hour
# FIX 10: Convert domestic hour data to DataFrame
domestic_hour = filtered[filtered["Domestic"] == True].groupby("hour").size().reindex(range(24), fill_value=0)
if domestic_hour.sum() > 0:
    domestic_hour_df = domestic_hour.reset_index()
    domestic_hour_df.columns = ["hour", "count"]
    fig_domestic_hour = px.line(
        domestic_hour_df,
        x="hour",
        y="count",
        markers=True,
        title="Domestic Crimes by Hour",
        labels={"hour": "Hour", "count": "Number of Domestic Crimes"}
    )
    fig_domestic_hour.update_xaxes(tickmode="linear")
    st.plotly_chart(fig_domestic_hour, use_container_width=True)

st.markdown(
    "Domestic incidents are often concentrated in private locations and may show different hourly patterns than the overall crime profile."
)

st.success("✅ Exploratory analysis dashboard loaded successfully!")
