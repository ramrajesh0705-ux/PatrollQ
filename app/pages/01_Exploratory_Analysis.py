import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Exploratory Data Analysis", page_icon="🔎", layout="wide")

st.title("🔎 Chicago Crime Exploratory Analysis")
st.markdown(
    "This dashboard provides interactive exploration of Chicago crime data. "
    "Explore crime patterns across Chicago."
)

@st.cache_data
def load_data():
    # Try multiple possible file locations
    possible_paths = [
        "data/processed/crime_cleaned.csv",
        "data/raw/Crimes_-_2001_to_Present.csv",
        "Crimes_-_2001_to_Present.csv",
        "chicago_crime.csv"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        st.error("❌ Crime data file not found. Please place the CSV file in the current directory.")
        st.info("You can download Chicago crime data from: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2")
        return None
    
    try:
        # Load data with low_memory=False to handle mixed types
        df = pd.read_csv(data_path, low_memory=False)
        st.success(f"✅ Raw data loaded: {len(df):,} rows, {len(df.columns)} columns")
        
        # Display column names for debugging
        with st.expander("🔍 Debug: Show column names"):
            st.write(df.columns.tolist())
        
        # --- FIX: Handle Chicago Crime Dataset specific columns ---
        # Check for different possible date column names
        date_column = None
        for col in ['Date', 'Occurred Date', 'Incident Date', 'Date of Occurrence']:
            if col in df.columns:
                date_column = col
                break
        
        if date_column is None:
            st.error(f"No date column found. Available columns: {df.columns.tolist()[:10]}...")
            return None
        
        st.info(f"Using date column: '{date_column}'")
        
        # Parse dates with Chicago dataset format (MM/DD/YYYY HH:MM:SS AM/PM)
        try:
            # First, convert to string and clean
            df['Date_parsed'] = pd.to_datetime(
                df[date_column], 
                format='%m/%d/%Y %I:%M:%S %p',  # Chicago format: 01/01/2023 12:05:00 PM
                errors='coerce'
            )
            
            # If that fails, try without time
            if df['Date_parsed'].isna().all():
                df['Date_parsed'] = pd.to_datetime(
                    df[date_column],
                    format='%m/%d/%Y',
                    errors='coerce'
                )
            
            # Final fallback - let pandas infer
            if df['Date_parsed'].isna().all():
                df['Date_parsed'] = pd.to_datetime(df[date_column], errors='coerce')
                
        except Exception as e:
            st.warning(f"Date parsing with specific format failed, trying automatic: {str(e)}")
            df['Date_parsed'] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Drop rows with invalid dates
        initial_len = len(df)
        df = df.dropna(subset=['Date_parsed'])
        st.info(f"🗑️ Dropped {initial_len - len(df):,} rows with invalid dates")
        
        if len(df) == 0:
            st.error("❌ No valid records after processing. Please check your data format.")
            # Show sample of date values for debugging
            with st.expander("Debug: Sample of original date values"):
                st.write(df[date_column].head(20).tolist())
            return None
        
        # Extract temporal features
        df['Year'] = df['Date_parsed'].dt.year
        df['Month'] = df['Date_parsed'].dt.month
        df['Day'] = df['Date_parsed'].dt.day
        df['Hour'] = df['Date_parsed'].dt.hour
        df['DayOfWeek'] = df['Date_parsed'].dt.dayofweek
        
        # Add time of day categorization
        def get_time_of_day(hour):
            if 0 <= hour < 5:
                return 'Late Night'
            elif 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Late Night'
        
        df['TimeOfDay'] = df['Hour'].apply(get_time_of_day)
        
        # Add weekend indicator
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6])  # Saturday=5, Sunday=6
        
        # Map crime types to severity levels
        severity_map = {
            'HOMICIDE': 'High',
            'ASSAULT': 'Medium',
            'BATTERY': 'Medium',
            'ROBBERY': 'High',
            'BURGLARY': 'Medium',
            'THEFT': 'Low',
            'MOTOR VEHICLE THEFT': 'Medium',
            'CRIMINAL DAMAGE': 'Low',
            'NARCOTICS': 'Low',
            'OTHER OFFENSE': 'Low'
        }
        
        if 'Primary Type' in df.columns:
            df['Primary Type'] = df['Primary Type'].astype(str).str.upper()
            df['CrimeSeverity'] = df['Primary Type'].map(severity_map).fillna('Low')
        
        # Convert numeric columns
        numeric_cols = ['Latitude', 'Longitude', 'District', 'Community Area', 'Beat', 'Ward']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert boolean columns
        if 'Arrest' in df.columns:
            df['Arrest'] = df['Arrest'].astype(str).str.upper().isin(['TRUE', '1', 'YES', 'T'])
        
        if 'Domestic' in df.columns:
            df['Domestic'] = df['Domestic'].astype(str).str.upper().isin(['TRUE', '1', 'YES', 'T'])
        
        st.success(f"✅ Successfully processed {len(df):,} crime records")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Load data
with st.spinner("Loading crime data... This may take a moment."):
    df = load_data()

if df is None or len(df) == 0:
    st.error("❌ Could not load valid crime data. Please check the file format.")
    st.stop()

# Show dataset info
st.markdown(f"**📊 Dataset loaded:** {len(df):,} records from {df['Year'].min()} to {df['Year'].max()}")

st.markdown("---")

# ============================================================================
# 1. CRIME DISTRIBUTION
# ============================================================================

st.header("📊 1. Crime Distribution Analysis")

if 'Primary Type' in df.columns:
    crime_counts = df['Primary Type'].value_counts()
    
    if len(crime_counts) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Overall Crime Distribution")
            fig_crime = px.bar(
                x=crime_counts.values,
                y=crime_counts.index,
                orientation='h',
                title="Crime Count by Primary Type",
                labels={'x': 'Number of Crimes', 'y': 'Crime Type'},
                color=crime_counts.values,
                color_continuous_scale='Blues',
                height=600
            )
            fig_crime.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_crime, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 Crimes")
            top10 = crime_counts.head(10)
            fig_top10 = px.bar(
                x=top10.values,
                y=top10.index,
                orientation='h',
                title="Top 10 Crime Types",
                labels={'x': 'Count', 'y': ''},
                color=top10.values,
                color_continuous_scale='Reds'
            )
            fig_top10.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig_top10, use_container_width=True)
            
            st.subheader("Rare Crimes (Bottom 10)")
            rare10 = crime_counts.tail(10)
            if len(rare10) > 0:
                fig_rare = px.bar(
                    x=rare10.values,
                    y=rare10.index,
                    orientation='h',
                    title="Least Frequent Crime Types",
                    labels={'x': 'Count', 'y': ''},
                    color=rare10.values,
                    color_continuous_scale='Greens'
                )
                fig_rare.update_layout(yaxis=dict(autorange="reversed"), height=400)
                st.plotly_chart(fig_rare, use_container_width=True)
    else:
        st.warning("No crime type data available.")
else:
    st.warning("'Primary Type' column not found in dataset.")

st.markdown("---")

# ============================================================================
# 2. CRIME SEVERITY ANALYSIS
# ============================================================================

if 'CrimeSeverity' in df.columns:
    st.header("⚠️ 2. Crime Severity Analysis")
    severity_counts = df['CrimeSeverity'].dropna().value_counts()
    if len(severity_counts) > 0:
        fig_severity = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title="Crime Distribution by Severity Level",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        st.plotly_chart(fig_severity, use_container_width=True)
    else:
        st.info("No severity data available.")
    st.markdown("---")

# ============================================================================
# 3. GEOGRAPHIC PATTERNS
# ============================================================================

st.header("🗺️ 3. Geographic Crime Patterns")

col1, col2 = st.columns(2)

with col1:
    if 'District' in df.columns:
        st.subheader("Crime by Police District")
        district_counts = df['District'].dropna().value_counts().sort_index()
        if len(district_counts) > 0:
            fig_dist = px.bar(
                x=district_counts.index.astype(str),
                y=district_counts.values,
                labels={'x': 'Police District', 'y': 'Number of Crimes'},
                title="Crime Distribution by District",
                color=district_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    if 'Community Area' in df.columns:
        st.subheader("Crime by Community Area")
        community_counts = df['Community Area'].dropna().value_counts().head(20)
        if len(community_counts) > 0:
            fig_comm = px.bar(
                x=community_counts.index.astype(str),
                y=community_counts.values,
                labels={'x': 'Community Area', 'y': 'Number of Crimes'},
                title="Top 20 Community Areas by Crime Count",
                color=community_counts.values,
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig_comm, use_container_width=True)

# Scatter map for crime locations
st.subheader("Crime Location Map")
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    map_data = df.dropna(subset=['Latitude', 'Longitude'])
    if len(map_data) > 0:
        # Sample for performance
        sample_size = min(10000, len(map_data))
        map_sample = map_data.sample(n=sample_size, random_state=42)
        
        fig_scatter = px.scatter_mapbox(
            map_sample,
            lat='Latitude',
            lon='Longitude',
            size_max=5,
            zoom=10,
            height=600,
            title=f"Crime Locations Map (Sampled: {sample_size:,} points)",
            mapbox_style="open-street-map",
            opacity=0.5,
            hover_data=['Primary Type'] if 'Primary Type' in df.columns else None
        )
        fig_scatter.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No valid location coordinates found in the dataset.")
else:
    st.info("Latitude/Longitude columns not available.")

st.markdown("---")

# ============================================================================
# 4. TEMPORAL TRENDS
# ============================================================================

st.header("📅 4. Temporal Crime Trends")

col1, col2 = st.columns(2)

with col1:
    if 'Year' in df.columns:
        st.subheader("Crimes by Year")
        year_counts = df['Year'].dropna().value_counts().sort_index()
        if len(year_counts) > 0:
            fig_year = px.line(
                x=year_counts.index,
                y=year_counts.values,
                markers=True,
                title="Yearly Crime Trends",
                labels={'x': 'Year', 'y': 'Number of Crimes'}
            )
            st.plotly_chart(fig_year, use_container_width=True)

with col2:
    if 'Month' in df.columns:
        st.subheader("Crimes by Month")
        month_counts = df['Month'].dropna().value_counts().sort_index()
        if len(month_counts) > 0:
            fig_month = px.bar(
                x=month_counts.index,
                y=month_counts.values,
                labels={'x': 'Month', 'y': 'Number of Crimes'},
                title="Monthly Crime Distribution",
                color=month_counts.values,
                color_continuous_scale='Teal'
            )
            st.plotly_chart(fig_month, use_container_width=True)

if 'Hour' in df.columns:
    st.subheader("Crime by Hour of Day")
    hour_counts = df['Hour'].dropna().value_counts().sort_index()
    if len(hour_counts) > 0:
        fig_hour = px.line(
            x=hour_counts.index,
            y=hour_counts.values,
            markers=True,
            title="Hourly Crime Patterns",
            labels={'x': 'Hour of Day', 'y': 'Number of Crimes'}
        )
        fig_hour.update_xaxes(tickmode='linear', dtick=2)
        st.plotly_chart(fig_hour, use_container_width=True)

if 'TimeOfDay' in df.columns:
    st.subheader("Time of Day Distribution")
    time_counts = df['TimeOfDay'].dropna().value_counts()
    time_order = ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Late Night']
    time_counts = time_counts.reindex([t for t in time_order if t in time_counts.index])
    if len(time_counts) > 0:
        fig_time = px.bar(
            x=time_counts.index,
            y=time_counts.values,
            labels={'x': 'Time of Day', 'y': 'Number of Crimes'},
            title="Crimes by Time Period",
            color=time_counts.values,
            color_continuous_scale='Purples'
        )
        st.plotly_chart(fig_time, use_container_width=True)

st.markdown("---")

# ============================================================================
# 5. WEEKEND VS WEEKDAY ANALYSIS
# ============================================================================

if 'IsWeekend' in df.columns:
    st.header("📆 5. Weekend vs Weekday Analysis")
    
    weekend_counts = df['IsWeekend'].value_counts()
    weekend_labels = {0: 'Weekday', 1: 'Weekend'}
    weekend_counts.index = [weekend_labels.get(i, str(i)) for i in weekend_counts.index]
    
    if len(weekend_counts) > 0:
        fig_weekend = px.pie(
            values=weekend_counts.values,
            names=weekend_counts.index,
            title="Weekday vs Weekend Crime Distribution",
            hole=0.4,
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        st.plotly_chart(fig_weekend, use_container_width=True)
    
    # Hourly comparison - weekday vs weekend
    if 'Hour' in df.columns:
        hourly_weekday = df[df['IsWeekend'] == 0].groupby('Hour').size()
        hourly_weekend = df[df['IsWeekend'] == 1].groupby('Hour').size()
        
        hourly_compare = pd.DataFrame({
            'Weekday': hourly_weekday.reindex(range(24), fill_value=0),
            'Weekend': hourly_weekend.reindex(range(24), fill_value=0)
        })
        
        fig_compare = px.line(
            hourly_compare,
            x=hourly_compare.index,
            y=['Weekday', 'Weekend'],
            markers=True,
            title="Hourly Crime Patterns: Weekday vs Weekend",
            labels={'x': 'Hour of Day', 'value': 'Number of Crimes', 'variable': 'Day Type'}
        )
        fig_compare.update_xaxes(tickmode='linear', dtick=2)
        st.plotly_chart(fig_compare, use_container_width=True)
    
    st.markdown("---")

# ============================================================================
# 6. ARREST AND DOMESTIC ANALYSIS
# ============================================================================

st.header("🚔 6. Arrest and Domestic Incident Analysis")

col1, col2 = st.columns(2)

with col1:
    if 'Arrest' in df.columns:
        st.subheader("Arrest Rate")
        arrest_counts = df['Arrest'].value_counts()
        arrest_labels = {True: 'Arrest Made', False: 'No Arrest'}
        arrest_counts.index = [arrest_labels.get(i, str(i)) for i in arrest_counts.index]
        if len(arrest_counts) > 0:
            fig_arrest = px.pie(
                values=arrest_counts.values,
                names=arrest_counts.index,
                title="Arrest vs No Arrest",
                hole=0.4,
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            st.plotly_chart(fig_arrest, use_container_width=True)

with col2:
    if 'Domestic' in df.columns:
        st.subheader("Domestic Incidents")
        domestic_counts = df['Domestic'].value_counts()
        domestic_labels = {True: 'Domestic', False: 'Non-Domestic'}
        domestic_counts.index = [domestic_labels.get(i, str(i)) for i in domestic_counts.index]
        if len(domestic_counts) > 0:
            fig_domestic = px.pie(
                values=domestic_counts.values,
                names=domestic_counts.index,
                title="Domestic vs Non-Domestic",
                hole=0.4,
                color_discrete_sequence=['#9b59b6', '#3498db']
            )
            st.plotly_chart(fig_domestic, use_container_width=True)

# Cross-tabulation for arrest rate by domestic status
if 'Arrest' in df.columns and 'Domestic' in df.columns:
    st.subheader("Arrest Rate by Domestic Incident Type")
    try:
        arrest_by_domestic = pd.crosstab(df['Domestic'], df['Arrest'], normalize='index') * 100
        arrest_by_domestic.columns = ['Not Arrested' if c is False else 'Arrested' for c in arrest_by_domestic.columns]
        arrest_by_domestic.index = ['Non-Domestic' if i is False else 'Domestic' for i in arrest_by_domestic.index]
        
        if len(arrest_by_domestic) > 0:
            fig_ad = px.bar(
                arrest_by_domestic.reset_index(),
                x='Domestic',
                y=['Arrested', 'Not Arrested'],
                title="Arrest Percentage by Domestic Incident Type",
                barmode='stack',
                labels={'value': 'Percentage', 'Domestic': 'Incident Type', 'variable': 'Status'}
            )
            st.plotly_chart(fig_ad, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not create cross-tabulation: {str(e)}")

st.markdown("---")

# ============================================================================
# 7. LOCATION TYPE ANALYSIS
# ============================================================================

if 'Location Description' in df.columns:
    st.header("🏢 7. Crime Location Analysis")
    
    st.subheader("Top Crime Locations")
    location_counts = df['Location Description'].value_counts().head(15)
    if len(location_counts) > 0:
        fig_location = px.bar(
            x=location_counts.values,
            y=location_counts.index,
            orientation='h',
            title="Top 15 Locations Where Crimes Occur",
            labels={'x': 'Number of Crim
