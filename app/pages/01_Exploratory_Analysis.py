import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="ðﾟﾔﾎ", layout="wide")

st.title("Chicago Crime Exploratory Analysis")
st.markdown(
    "This page converts the exploratory notebook into an interactive Streamlit dashboard. "
    "Use the controls to filter the dataset and explore crime patterns by type, geography, time, arrest status, and domestic incidents."
)

@st.cache_data
def load_data():
    data_path = "data/processed/crime_cleaned.csv"
    if not os.path.exists(data_path):
        return None
    df = pd.read_csv(data_path)
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

    df["season"] = df["Month"].apply(get_season)
    return df


df = load_data()
if df is None:
    st.error("❌ Could not find preprocesed_data.csv in the app root.")
    st.stop()

st.markdown(f"**Dataset loaded:** {len(df):,} records")

with st.sidebar:
    st.header("Filters")
    sample_size = st.slider("Sample size for maps and global charts", 10000, min(50000, len(df)), min(200000, len(df)), step=10000)
    year_options = [int(y) for y in sorted(df["Year"].dropna().unique())]
    selected_year = st.selectbox("Year", ["All"] + year_options, index=0)
    selected_day = st.selectbox("Day of week", ["All"] + ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=0)
    selected_season = st.selectbox("Season", ["All", "Winter", "Spring", "Summer", "Fall"], index=0)

filtered = df.copy()
if selected_year != "All":
    filtered = filtered[filtered["Year"] == int(selected_year)]
if selected_day != "All":
    filtered = filtered[filtered["day_name"] == selected_day]
if selected_season != "All":
    filtered = filtered[filtered["season"] == selected_season]

st.markdown("---")
st.header("list of columns")
st.write(filtered.columns)
st.markdown("----")
# Crime Distribution
st.header("1. Crime Distribution")

crime_counts = filtered["Primary Type"].value_counts().sort_values(ascending=False)
if len(crime_counts) == 0:
    st.warning("No records match the selected filters.")
    st.stop()

st.subheader("Crime frequency by type")
fig_crime = px.bar(
    x=crime_counts.values,
    y=crime_counts.index,
    orientation="h",
    title="Crime Count by Primary Type",
    labels={"x": "Count", "y": "Primary Type"},
    color=crime_counts.values,
    color_continuous_scale="Blues"
)
fig_crime.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_crime, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 10 Crime Types")
    top10 = crime_counts.head(10)
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
    rare10 = crime_counts.tail(10)
    fig_rare = px.bar(
        x=rare10.values,
        y=rare10.index,
        orientation="h",
        labels={"x": "Count", "y": "Crime Type"},
        title="Rare Crime Types"
    )
    fig_rare.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_rare, use_container_width=True)

st.markdown(
    "Crimes are concentrated among a small number of categories, with property-related offenses like THEFT and BATTERY dominating the dataset."
)

st.markdown("---")

# Geographic Patterns
st.header("2. Geographic Crime Patterns")

st.subheader("Crime distribution by police district and community area")

district_counts = filtered["District"].value_counts().sort_index()
community_counts = filtered["Community Area"].value_counts().sort_index()
col1, col2 = st.columns(2)
with col1:
    fig_dist = px.bar(
        x=district_counts.index.astype(str),
        y=district_counts.values,
        labels={"x": "District", "y": "Number of Crimes"},
        title="Crime by Police District",
    )
    st.plotly_chart(fig_dist, use_container_width=True)
with col2:
    fig_comm = px.bar(
        x=community_counts.index.astype(str),
        y=community_counts.values,
        labels={"x": "Community Area", "y": "Number of Crimes"},
        title="Crime by Community Area",
    )
    st.plotly_chart(fig_comm, use_container_width=True)

map_sample = filtered.dropna(subset=["Latitude", "Longitude"]).sample(n=10000, random_state=42)
if len(map_sample) > 0:
    fig_map = px.density_mapbox(
        map_sample,
        lat="Latitude",
        lon="Longitude",
        radius=10,
        zoom=10,
        height=600,
        title="Crime Density Map",
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("No latitude/longitude data is available for the selected filters.")

st.markdown("---")

# Temporal Trends
st.header("3. Temporal Crime Trends")

st.subheader("Total crimes by year")
crimes_per_year = filtered["Year"].value_counts().sort_index()
fig_year = px.line(
    x=crimes_per_year.index,
    y=crimes_per_year.values,
    markers=True,
    title="Total Crimes Per Year",
    labels={"x": "Year", "y": "Crime Count"}
)
st.plotly_chart(fig_year, use_container_width=True)

st.subheader("Crimes by month")
import calendar

# 1. Create a list of month abbreviations: ['Jan', 'Feb', 'Mar', ..., 'Dec']
# (We use [1:] because calendar.month_abbr[0] is an empty string)
month_names = list(calendar.month_abbr)[1:]

# 2. Count crimes and reindex using the month names
crimes_by_month = filtered["Month"].value_counts().reindex(range(1, 13), fill_value=0)
crimes_by_month.index = month_names

# 3. Plot exactly as you did before
fig_month = px.bar(
    x=crimes_by_month.index,
    y=crimes_by_month.values,
    labels={"x": "Month", "y": "Number of Crimes"},
    title="Crime Volume by Month"
)
st.plotly_chart(fig_month, use_container_width=True)

st.subheader("Year-month crime heatmap")
# Create list of month names: ['Jan', 'Feb', ..., 'Dec']
month_names = list(calendar.month_abbr)[1:]

crimes_per_month = filtered.groupby(["Year", "Month"]).size().reset_index(name="count")

if not crimes_per_month.empty:
    # FIX: Changed columns="month" to columns="Month" to match your groupby casing
    heat = crimes_per_month.pivot(index="Year", columns="Month", values="count").fillna(0)
    
    # Reindex to force columns 1 through 12, then rename to Jan-Dec
    heat = heat.reindex(columns=range(1, 13), fill_value=0)
    heat.columns = month_names

    fig_heat = go.Figure(data=go.Heatmap(
        z=heat.values,
        x=heat.columns,   # Now passes ['Jan', 'Feb', ..., 'Dec']
        y=heat.index,
        colorscale="Viridis",
        colorbar=dict(title="Count")
    ))
    fig_heat.update_layout(
        title="Crime Trends by Year and Month",
        xaxis_title="Month",
        yaxis_title="Year"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.warning("Not enough data to build the year-month heatmap.")

st.subheader("Crimes by day of week")
# Lists for reindexing and renaming
days_long = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
days_short = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
day_of_week_dict = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday"
}
# Map the numeric DayOfWeek to day names
filtered['DayOfWeek'] = filtered['DayOfWeek'].map(day_of_week_dict)

# Count and reindex using full names
crime_by_day = filtered["DayOfWeek"].value_counts().reindex(days_long, fill_value=0)
# Replace full names with 3-letter abbreviations
crime_by_day.index = days_short

fig_day = px.bar(
    x=crime_by_day.index,
    y=crime_by_day.values,
    labels={"x": "Day of Week", "y": "Number of Crimes"},
    title="Crime Counts by Day of Week"
)
st.plotly_chart(fig_day, use_container_width=True)

st.subheader("Crimes by hour of day")
crime_by_hour = filtered["Hour"].value_counts().sort_index()
fig_hour = px.line(
    x=crime_by_hour.index,
    y=crime_by_hour.values,
    markers=True,
    title="Crime Patterns by Hour",
    labels={"x": "Hour of Day", "y": "Crime Count"}
)
fig_hour.update_xaxes(tickmode="linear")
st.plotly_chart(fig_hour, use_container_width=True)

st.markdown("---")

# Season Analysis
st.header("4. Seasonal Crime Patterns")
season_counts = filtered["season"].value_counts().reindex(["Winter", "Spring", "Summer", "Fall"], fill_value=0)
fig_season = px.bar(
    x=season_counts.index,
    y=season_counts.values,
    labels={"x": "Season", "y": "Number of Crimes"},
    title="Crime Count by Season"
)
st.plotly_chart(fig_season, use_container_width=True)

st.markdown(
    "Seasonal analysis shows whether crime volume increases during warm months or dips in winter. "
    "This helps identify seasonal police resource planning needs."
)

st.markdown("---")

# Arrest and Domestic Analysis
st.header("5. Arrest and Domestic Incident Analysis")

arrest_counts = filtered["Arrest"].value_counts().rename(index={True: "Arrested", False: "Not Arrested"})
arrest_pct = (arrest_counts / arrest_counts.sum() * 100).round(2)

domestic_counts = filtered["Domestic"].value_counts().rename(index={True: "Domestic", False: "Non-Domestic"})
domestic_pct = (domestic_counts / domestic_counts.sum() * 100).round(2)

col1, col2 = st.columns(2)
with col1:
    fig_arrest = px.pie(
        names=arrest_pct.index,
        values=arrest_pct.values,
        title="Arrest Rate Distribution",
        hole=0.4,
    )
    st.plotly_chart(fig_arrest, use_container_width=True)
with col2:
    fig_domestic = px.pie(
        names=domestic_pct.index,
        values=domestic_pct.values,
        title="Domestic vs Non-Domestic Distribution",
        hole=0.4,
    )
    st.plotly_chart(fig_domestic, use_container_width=True)

st.subheader("Arrest rate by domestic incident")
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
else:
    st.warning("Not enough data to calculate arrest rate by domestic flag.")

st.subheader("Domestic crimes by hour")
domestic_hour = filtered[filtered["Domestic"] == True].groupby("Hour").size().reindex(range(24), fill_value=0)
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
