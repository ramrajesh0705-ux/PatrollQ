import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="🔎", layout="wide")

st.title("🔎 Chicago Crime Exploratory Analysis")
st.markdown(
    "This page converts the exploratory notebook into an interactive Streamlit dashboard. "
    "Use the controls to filter the dataset and explore crime patterns by type, geography, time, arrest status, and domestic incidents."
)

@st.cache_data
def load_data():
    """Load and preprocess data with optimized dtypes and error handling"""
    data_path = "data/processed/crime_cleaned.csv"
    if not os.path.exists(data_path):
        alt_path = "crime_cleaned.csv"
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            return None
    
    try:
        df = pd.read_csv(data_path, low_memory=False)
        
        # Check if required columns exist
        required_columns = ["Date", "Primary Type", "Arrest", "Domestic", "Latitude", "Longitude"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Convert Date column safely
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y %H:%M", errors="coerce")
            df = df.dropna(subset=["Date"])
        
        # Convert numeric columns safely
        if "Latitude" in df.columns:
            df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        if "Longitude" in df.columns:
            df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        
        # Convert boolean columns safely
        if "Arrest" in df.columns:
            df["Arrest"] = df["Arrest"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
            df["Arrest"] = df["Arrest"].astype("boolean")
        
        if "Domestic" in df.columns:
            df["Domestic"] = df["Domestic"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
            df["Domestic"] = df["Domestic"].astype("boolean")
        
        # Convert categorical columns
        if "Primary Type" in df.columns:
            df["Primary Type"] = df["Primary Type"].astype("category")
        
        # Handle District and Community Area if they exist
        if "District" in df.columns:
            df["District"] = pd.to_numeric(df["District"], errors="coerce").astype("Int64")
        if "Community Area" in df.columns:
            df["Community Area"] = pd.to_numeric(df["Community Area"], errors="coerce").astype("Int64")
        
        # Add derived columns
        if "Date" in df.columns:
            df["month"] = df["Date"].dt.month
            df["hour"] = df["Date"].dt.hour
            df["day_name"] = df["Date"].dt.day_name()
            df["Year"] = df["Date"].dt.year
        
        # Define season function
        def get_season(month: int) -> str:
            if pd.isna(month):
                return "Unknown"
            if month in [12, 1, 2]:
                return "Winter"
            if month in [3, 4, 5]:
                return "Spring"
            if month in [6, 7, 8]:
                return "Summer"
            return "Fall"
        
        if "month" in df.columns:
            df["season"] = df["month"].apply(get_season).astype("category")
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
df = load_data()
if df is None:
    st.error("❌ Could not find crime_cleaned.csv in the data/processed/ directory. Please check file path.")
    st.stop()

st.markdown(f"**Dataset loaded:** {len(df):,} records")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    
    # Initialize session state
    if 'sample_size' not in st.session_state:
        st.session_state.sample_size = min(50000, len(df))
    
    # FIXED: Handle case where dataset is too small for slider
    total_records = len(df)
    max_sample_default = min(50000, total_records)
    
    # Calculate valid min and max for slider
    slider_min = 1000  # Minimum sample size
    slider_max = max(slider_min, max_sample_default)  # Ensure max >= min
    
    # FIXED: Ensure we don't try to create slider with invalid range
    if slider_max > slider_min:
        sample_size = st.slider(
            "Sample size for maps", 
            slider_min, 
            slider_max, 
            min(st.session_state.sample_size, slider_max), 
            step=max(1000, (slider_max - slider_min) // 10),  # Dynamic step size
            help=f"Larger samples show more detail but take longer to render (max: {slider_max:,} records)"
        )
    else:
        # If dataset is too small, just show a info message and use all data
        sample_size = slider_max
        st.info(f"Dataset has only {total_records:,} records. Using all available data for maps.")
    
    st.session_state.sample_size = sample_size
    
    # Year filter
    year_options = []
    if "Year" in df.columns:
        year_options = sorted([int(y) for y in df["Year"].dropna().unique()])
    
    selected_year = st.selectbox("Year", ["All"] + year_options, index=0) if year_options else "All"
    
    # Day filter
    selected_day = st.selectbox(
        "Day of week", 
        ["All"] + ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 
        index=0
    )
    
    # Season filter
    selected_season = st.selectbox("Season", ["All", "Winter", "Spring", "Summer", "Fall"], index=0)

# Apply filters efficiently
@st.cache_data
def apply_filters(_df, year, day, season):
    """Apply filters with caching and validation"""
    try:
        mask = pd.Series([True] * len(_df), index=_df.index)
        
        if year != "All" and "Year" in _df.columns:
            mask &= (_df["Year"] == int(year))
        
        if day != "All" and "day_name" in _df.columns:
            mask &= (_df["day_name"] == day)
        
        if season != "All" and "season" in _df.columns:
            mask &= (_df["season"] == season)
        
        filtered_df = _df[mask]
        return filtered_df
    except Exception as e:
        st.error(f"Error applying filters: {str(e)}")
        return _df

filtered = apply_filters(df, selected_year, selected_day, selected_season)

# Check if filtered data is empty
if len(filtered) == 0:
    st.warning("⚠️ No records match the selected filters. Please adjust your filter criteria.")
    st.info("Try selecting 'All' for some filters to see more data.")
    st.stop()

st.markdown("---")

# Helper function to safely create charts
def safe_create_chart(chart_func, fallback_message="No data available for this chart"):
    """Safely create charts with error handling"""
    try:
        return chart_func()
    except Exception as e:
        st.warning(f"{fallback_message}: {str(e)}")
        return None

# 1. Crime Distribution
st.header("1. Crime Distribution")

if "Primary Type" not in filtered.columns:
    st.error("Primary Type column not found in data")
    st.stop()

filtered_crime_counts = filtered["Primary Type"].value_counts()

if len(filtered_crime_counts) == 0:
    st.warning("No crime types found in filtered data")
else:
    st.subheader("Crime frequency by type")
    
    def create_crime_bar():
        fig_crime = px.bar(
            x=filtered_crime_counts.values,
            y=filtered_crime_counts.index,
            orientation="h",
            title="Crime Count by Primary Type",
            labels={"x": "Count", "y": "Primary Type"},
            color=filtered_crime_counts.values,
            color_continuous_scale="Blues"
        )
        fig_crime.update_layout(yaxis=dict(autorange="reversed"), height=600)
        return fig_crime
    
    safe_create_chart(create_crime_bar, "Could not create crime distribution chart")
    st.plotly_chart(fig_crime if 'fig_crime' in locals() else None, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Crime Types")
        top10 = filtered_crime_counts.head(10)
        if len(top10) > 0:
            fig_top10 = px.bar(
                x=top10.values,
                y=top10.index,
                orientation="h",
                labels={"x": "Count", "y": "Crime Type"},
                title="Top 10 Crime Types"
            )
            fig_top10.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_top10, use_container_width=True)
    
    with col2:
        st.subheader("Least Frequent Crime Types")
        non_zero_counts = filtered_crime_counts[filtered_crime_counts > 0]
        rare10 = non_zero_counts.tail(10) if len(non_zero_counts) > 0 else pd.Series()
        if len(rare10) > 0:
            fig_rare = px.bar(
                x=rare10.values,
                y=rare10.index,
                orientation="h",
                labels={"x": "Count", "y": "Crime Type"},
                title="Rare Crime Types"
            )
            fig_rare.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_rare, use_container_width=True)

st.markdown("---")

# 2. Geographic Patterns
st.header("2. Geographic Crime Patterns")

if "District" in filtered.columns:
    district_counts = filtered["District"].value_counts().sort_index()
    if len(district_counts) > 0:
        fig_dist = px.bar(
            x=district_counts.index.astype(str),
            y=district_counts.values,
            labels={"x": "District", "y": "Number of Crimes"},
            title="Crime by Police District",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

if "Community Area" in filtered.columns:
    community_counts = filtered["Community Area"].value_counts().sort_index()
    if len(community_counts) > 0:
        top_communities = community_counts.head(20)
        fig_comm = px.bar(
            x=top_communities.index.astype(str),
            y=top_communities.values,
            labels={"x": "Community Area", "y": "Number of Crimes"},
            title="Crime by Community Area (Top 20)",
        )
        st.plotly_chart(fig_comm, use_container_width=True)

# Maps
st.subheader("Spatial density and location clustering")

if "Latitude" in filtered.columns and "Longitude" in filtered.columns:
    map_data = filtered.dropna(subset=["Latitude", "Longitude"])
    
    if len(map_data) > 0:
        # FIXED: Ensure sample size doesn't exceed available data
        actual_sample_size = min(sample_size, len(map_data))
        
        # FIXED: Only sample if needed
        if len(map_data) > actual_sample_size:
            map_sample = map_data.sample(n=actual_sample_size, random_state=42)
        else:
            map_sample = map_data
        
        # Density map with error handling
        try:
            fig_map = px.density_mapbox(
                map_sample,
                lat="Latitude",
                lon="Longitude",
                radius=8,
                zoom=10,
                height=500,
                title=f"Crime Density Map (Sample: {len(map_sample):,} points)",
            )
            fig_map.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create density map: {str(e)}")
        
        # Cluster map
        with st.spinner("Generating cluster map..."):
            try:
                map_sample_copy = map_sample.copy()
                map_sample_copy["lat_bin"] = map_sample_copy["Latitude"].round(3)
                map_sample_copy["lon_bin"] = map_sample_copy["Longitude"].round(3)
                grid_counts = map_sample_copy.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="crime_count")
                
                if len(grid_counts) > 2000:
                    grid_counts = grid_counts.nlargest(2000, "crime_count")
                
                if len(grid_counts) > 0:
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
            except Exception as e:
                st.warning(f"Could not create cluster map: {str(e)}")
    else:
        st.warning("No valid latitude/longitude data available for the selected filters.")

st.markdown("---")

# 3. Temporal Trends
st.header("3. Temporal Crime Trends")

if "Year" in filtered.columns and len(filtered["Year"].dropna()) > 0:
    crimes_per_year = filtered["Year"].value_counts().sort_index()
    if len(crimes_per_year) > 0:
        fig_year = px.line(
            x=crimes_per_year.index,
            y=crimes_per_year.values,
            markers=True,
            title="Total Crimes Per Year",
            labels={"x": "Year", "y": "Crime Count"}
        )
        st.plotly_chart(fig_year, use_container_width=True)

if "month" in filtered.columns:
    crimes_by_month = filtered["month"].value_counts().reindex(range(1, 13), fill_value=0)
    if crimes_by_month.sum() > 0:
        fig_month = px.bar(
            x=crimes_by_month.index,
            y=crimes_by_month.values,
            labels={"x": "Month", "y": "Number of Crimes"},
            title="Crime Volume by Month"
        )
        st.plotly_chart(fig_month, use_container_width=True)

# Year-month heatmap
if "Year" in filtered.columns and "month" in filtered.columns:
    year_month_data = filtered.groupby(["Year", "month"]).size().reset_index(name="count")
    if not year_month_data.empty and len(year_month_data["Year"].unique()) > 1:
        try:
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
        except Exception as e:
            st.warning(f"Could not create heatmap: {str(e)}")

# Day of week patterns
if "day_name" in filtered.columns:
    crime_by_day = filtered["day_name"].value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fill_value=0
    )
    if crime_by_day.sum() > 0:
        fig_day = px.bar(
            x=crime_by_day.index,
            y=crime_by_day.values,
            labels={"x": "Day of Week", "y": "Number of Crimes"},
            title="Crime Counts by Day of Week"
        )
        st.plotly_chart(fig_day, use_container_width=True)

# Hour patterns
if "hour" in filtered.columns:
    crime_by_hour = filtered["hour"].value_counts().sort_index()
    if crime_by_hour.sum() > 0:
        fig_hour = px.line(
            x=crime_by_hour.index,
            y=crime_by_hour.values,
            markers=True,
            title="Crime Patterns by Hour",
            labels={"x": "Hour of Day", "y": "Crime Count"}
        )
        fig_hour.update_xaxes(tickmode="linear")
        st.plotly_chart(fig_hour, use_container_width=True)

# Hourly patterns by year
if "Year" in filtered.columns and "hour" in filtered.columns:
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
if "season" in filtered.columns:
    season_counts = filtered["season"].value_counts().reindex(["Winter", "Spring", "Summer", "Fall"], fill_value=0)
    if season_counts.sum() > 0:
        fig_season = px.bar(
            x=season_counts.index,
            y=season_counts.values,
            labels={"x": "Season", "y": "Number of Crimes"},
            title="Crime Count by Season"
        )
        st.plotly_chart(fig_season, use_container_width=True)

st.markdown("---")

# 5. Arrest and Domestic Analysis
st.header("5. Arrest and Domestic Incident Analysis")

if "Arrest" in filtered.columns:
    arrest_counts = filtered["Arrest"].value_counts(dropna=False)
    if len(arrest_counts) > 0 and arrest_counts.sum() > 0:
        arrest_pct = (arrest_counts / arrest_counts.sum() * 100).round(2)
        arrest_labels = {True: "Arrested", False: "Not Arrested"}
        arrest_pct.index = [arrest_labels.get(x, str(x)) for x in arrest_pct.index]
        
        fig_arrest = px.pie(
            names=arrest_pct.index,
            values=arrest_pct.values,
            title=f"Arrest Rate: {arrest_pct.get('Arrested', 0):.1f}%",
            hole=0.4,
        )
        st.plotly_chart(fig_arrest, use_container_width=True)

if "Domestic" in filtered.columns:
    domestic_counts = filtered["Domestic"].value_counts(dropna=False)
    if len(domestic_counts) > 0 and domestic_counts.sum() > 0:
        domestic_pct = (domestic_counts / domestic_counts.sum() * 100).round(2)
        domestic_labels = {True: "Domestic", False: "Non-Domestic"}
        domestic_pct.index = [domestic_labels.get(x, str(x)) for x in domestic_pct.index]
        
        fig_domestic = px.pie(
            names=domestic_pct.index,
            values=domestic_pct.values,
            title=f"Domestic Incidents: {domestic_pct.get('Domestic', 0):.1f}%",
            hole=0.4,
        )
        st.plotly_chart(fig_domestic, use_container_width=True)

# Arrest rate by domestic incident
if "Arrest" in filtered.columns and "Domestic" in filtered.columns:
    try:
        arrest_domestic = pd.crosstab(filtered["Domestic"], filtered["Arrest"], normalize="index") * 100
        if not arrest_domestic.empty and len(arrest_domestic) > 0:
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
    except Exception as e:
        st.warning(f"Could not calculate arrest rates: {str(e)}")

# Domestic crimes by hour
if "Domestic" in filtered.columns and "hour" in filtered.columns:
    domestic_crimes = filtered[filtered["Domestic"] == True]
    if len(domestic_crimes) > 0:
        domestic_hour = domestic_crimes.groupby("hour").size().reindex(range(24), fill_value=0)
        if domestic_hour.sum() > 0:
            fig_domestic_hour = px.line(
                x=domestic_hour.index,
                y=domestic_hour.values,
                markers=True,
                title="Domestic Crimes by Hour",
                labels={"x": "Hour", "y": "Number of Domestic Crimes"}
            )
            fig_domestic_hour.update_xaxes(tickmode="linear")
            st.plotly_chart(fig_domestic_hour, use_container_width=True)

st.markdown(
    "Domestic incidents are often concentrated in private locations and may show different hourly patterns than the overall crime profile."
)

st.success("✅ Exploratory analysis dashboard loaded successfully!")
