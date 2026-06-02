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
    #data_path = "data/processed/crime_cleaned.csv"
    
    df = pd.read_csv("data/processed/crime_cleaned.csv")
    
    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
    
    # Convert numeric columns
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    # Convert boolean columns
    df['Arrest'] = df['Arrest'].astype(str).str.upper().map({'TRUE': True, 'FALSE': False, '1': True, '0': False})
    df['Domestic'] = df['Domestic'].astype(str).str.upper().map({'TRUE': True, 'FALSE': False, '1': True, '0': False})
    
    # Ensure numeric columns are proper types
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
    df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
    df['District'] = pd.to_numeric(df['District'], errors='coerce')
    df['Community Area'] = pd.to_numeric(df['Community Area'], errors='coerce')
    df['Beat'] = pd.to_numeric(df['Beat'], errors='coerce')
    df['Ward'] = pd.to_numeric(df['Ward'], errors='coerce')
    
    # Categorical columns for efficiency
    df['Primary Type'] = df['Primary Type'].astype('category')
    df['DayOfWeek'] = df['DayOfWeek'].astype('category')
    df['TimeOfDay'] = df['TimeOfDay'].astype('category')
    df['CrimeSeverity'] = df['CrimeSeverity'].astype('category')
    
    # Drop rows with null dates
    df = df.dropna(subset=['Date'])
    
    return df


# Load data with progress indicator
with st.spinner("Loading crime data... This may take a moment."):
    df = load_data()

if df is None:
    st.error("❌ Could not find crime_cleaned.csv in data/processed/ directory.")
    st.stop()

# Show dataset info
st.markdown(f"**Dataset loaded:** {len(df):,} records (first 500,000 rows)")

st.markdown("---")

# ============================================================================
# 1. CRIME DISTRIBUTION
# ============================================================================

st.header("📊 1. Crime Distribution Analysis")

crime_counts = df['Primary Type'].value_counts()

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

st.markdown("---")

# ============================================================================
# 2. CRIME SEVERITY ANALYSIS
# ============================================================================

st.header("⚠️ 2. Crime Severity Analysis")

severity_counts = df['CrimeSeverity'].value_counts()
fig_severity = px.pie(
    values=severity_counts.values,
    names=severity_counts.index,
    title="Crime Distribution by Severity Level",
    hole=0.4,
    color_discrete_sequence=px.colors.sequential.Reds_r
)
st.plotly_chart(fig_severity, use_container_width=True)

st.markdown("---")

# ============================================================================
# 3. GEOGRAPHIC PATTERNS
# ============================================================================

st.header("🗺️ 3. Geographic Crime Patterns")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Crime by Police District")
    district_counts = df['District'].value_counts().sort_index()
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
    st.subheader("Crime by Community Area")
    community_counts = df['Community Area'].value_counts().head(20)
    fig_comm = px.bar(
        x=community_counts.index.astype(str),
        y=community_counts.values,
        labels={'x': 'Community Area', 'y': 'Number of Crimes'},
        title="Top 20 Community Areas by Crime Count",
        color=community_counts.values,
        color_continuous_scale='Plasma'
    )
    st.plotly_chart(fig_comm, use_container_width=True)

# Scatter map for crime locations (replacing density map)
st.subheader("Crime Location Map")
map_data = df.dropna(subset=['Latitude', 'Longitude']).sample(n=min(10000, len(df)), random_state=42)

if len(map_data) > 100:
    fig_scatter = px.scatter_mapbox(
        map_data,
        lat='Latitude',
        lon='Longitude',
        color='Primary Type',
        size_max=8,
        zoom=10,
        height=600,
        title="Crime Locations Map (Sampled)",
        mapbox_style="open-street-map",
        opacity=0.6,
        hover_data=['Primary Type', 'Description', 'District']
    )
    fig_scatter.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Insufficient location data for map visualization.")

st.markdown("---")

# ============================================================================
# 4. TEMPORAL TRENDS
# ============================================================================

st.header("📅 4. Temporal Crime Trends")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Crimes by Year")
    year_counts = df['Year'].value_counts().sort_index()
    fig_year = px.line(
        x=year_counts.index,
        y=year_counts.values,
        markers=True,
        title="Yearly Crime Trends",
        labels={'x': 'Year', 'y': 'Number of Crimes'}
    )
    st.plotly_chart(fig_year, use_container_width=True)

with col2:
    st.subheader("Crimes by Month")
    month_counts = df['Month'].value_counts().sort_index()
    fig_month = px.bar(
        x=month_counts.index,
        y=month_counts.values,
        labels={'x': 'Month', 'y': 'Number of Crimes'},
        title="Monthly Crime Distribution",
        color=month_counts.values,
        color_continuous_scale='Teal'
    )
    st.plotly_chart(fig_month, use_container_width=True)

st.subheader("Crime by Hour of Day")
hour_counts = df['Hour'].value_counts().sort_index()
fig_hour = px.line(
    x=hour_counts.index,
    y=hour_counts.values,
    markers=True,
    title="Hourly Crime Patterns",
    labels={'x': 'Hour of Day', 'y': 'Number of Crimes'}
)
fig_hour.update_xaxes(tickmode='linear', dtick=2)
st.plotly_chart(fig_hour, use_container_width=True)

st.subheader("Time of Day Distribution")
time_counts = df['TimeOfDay'].value_counts()
time_order = ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Late Night']
time_counts = time_counts.reindex([t for t in time_order if t in time_counts.index])
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

st.header("📆 5. Weekend vs Weekday Analysis")

weekend_counts = df['IsWeekend'].value_counts()
weekend_labels = {0: 'Weekday', 1: 'Weekend'}
weekend_counts.index = [weekend_labels.get(i, str(i)) for i in weekend_counts.index]

fig_weekend = px.pie(
    values=weekend_counts.values,
    names=weekend_counts.index,
    title="Weekday vs Weekend Crime Distribution",
    hole=0.4,
    color_discrete_sequence=['#3498db', '#e74c3c']
)
st.plotly_chart(fig_weekend, use_container_width=True)

# Hourly comparison - weekday vs weekend
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
    st.subheader("Arrest Rate")
    arrest_counts = df['Arrest'].value_counts()
    arrest_labels = {True: 'Arrest Made', False: 'No Arrest'}
    arrest_counts.index = [arrest_labels.get(i, str(i)) for i in arrest_counts.index]
    fig_arrest = px.pie(
        values=arrest_counts.values,
        names=arrest_counts.index,
        title="Arrest vs No Arrest",
        hole=0.4,
        color_discrete_sequence=['#2ecc71', '#e74c3c']
    )
    st.plotly_chart(fig_arrest, use_container_width=True)

with col2:
    st.subheader("Domestic Incidents")
    domestic_counts = df['Domestic'].value_counts()
    domestic_labels = {True: 'Domestic', False: 'Non-Domestic'}
    domestic_counts.index = [domestic_labels.get(i, str(i)) for i in domestic_counts.index]
    fig_domestic = px.pie(
        values=domestic_counts.values,
        names=domestic_counts.index,
        title="Domestic vs Non-Domestic",
        hole=0.4,
        color_discrete_sequence=['#9b59b6', '#3498db']
    )
    st.plotly_chart(fig_domestic, use_container_width=True)

# Cross-tabulation for arrest rate by domestic status
st.subheader("Arrest Rate by Domestic Incident Type")
arrest_by_domestic = pd.crosstab(df['Domestic'], df['Arrest'], normalize='index') * 100
arrest_by_domestic.columns = ['Not Arrested' if c is False else 'Arrested' for c in arrest_by_domestic.columns]
arrest_by_domestic.index = ['Non-Domestic' if i is False else 'Domestic' for i in arrest_by_domestic.index]

fig_ad = px.bar(
    arrest_by_domestic.reset_index(),
    x='Domestic',
    y=['Arrested', 'Not Arrested'],
    title="Arrest Percentage by Domestic Incident Type",
    barmode='stack',
    labels={'value': 'Percentage', 'Domestic': 'Incident Type', 'variable': 'Status'}
)
st.plotly_chart(fig_ad, use_container_width=True)

# Domestic incidents by time of day
st.subheader("Domestic Incidents by Time of Day")
domestic_by_time = df[df['Domestic'] == True]['TimeOfDay'].value_counts()
domestic_by_time = domestic_by_time.reindex([t for t in time_order if t in domestic_by_time.index])
fig_domestic_time = px.bar(
    x=domestic_by_time.index,
    y=domestic_by_time.values,
    labels={'x': 'Time of Day', 'y': 'Number of Domestic Incidents'},
    title="Domestic Incidents Distribution by Time Period",
    color=domestic_by_time.values,
    color_continuous_scale='Oranges'
)
st.plotly_chart(fig_domestic_time, use_container_width=True)

st.markdown("---")

# ============================================================================
# 7. LOCATION TYPE ANALYSIS
# ============================================================================

st.header("🏢 7. Crime Location Analysis")

st.subheader("Top Crime Locations")
location_counts = df['Location Description'].value_counts().head(15)
fig_location = px.bar(
    x=location_counts.values,
    y=location_counts.index,
    orientation='h',
    title="Top 15 Locations Where Crimes Occur",
    labels={'x': 'Number of Crimes', 'y': 'Location Description'},
    color=location_counts.values,
    color_continuous_scale='Viridis'
)
fig_location.update_layout(yaxis=dict(autorange="reversed"), height=500)
st.plotly_chart(fig_location, use_container_width=True)

st.markdown("---")

# ============================================================================
# 8. CRIME SEVERITY BY LOCATION
# ============================================================================

st.header("📍 8. Crime Severity by Location Type")

# Sample for performance
location_severity_sample = df.groupby(['Location Description', 'CrimeSeverity']).size().reset_index(name='count')
top_locations = df['Location Description'].value_counts().head(10).index
location_severity_top = location_severity_sample[location_severity_sample['Location Description'].isin(top_locations)]

if len(location_severity_top) > 0:
    fig_severity_loc = px.bar(
        location_severity_top,
        x='Location Description',
        y='count',
        color='CrimeSeverity',
        title="Crime Severity Distribution by Location Type (Top 10 Locations)",
        labels={'count': 'Number of Crimes', 'Location Description': 'Location'},
        barmode='stack'
    )
    fig_severity_loc.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_severity_loc, use_container_width=True)
else:
    st.info("Insufficient data for severity by location analysis.")

st.markdown("---")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

st.header("📈 9. Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Unique Crime Types", df['Primary Type'].nunique())

with col3:
    arrest_rate = (df['Arrest'].sum() / len(df) * 100) if len(df) > 0 else 0
    st.metric("Overall Arrest Rate", f"{arrest_rate:.1f}%")

with col4:
    domestic_rate = (df['Domestic'].sum() / len(df) * 100) if len(df) > 0 else 0
    st.metric("Domestic Incident Rate", f"{domestic_rate:.1f}%")

st.success("✅ Exploratory analysis dashboard loaded successfully!")
