# app/pages/01_Crime_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
st.set_page_config(page_title="Crime Analysis", page_icon="📊", layout="wide")

st.title("📊 Crime Analysis Dashboard")

@st.cache_data
def load_data():
    try:
        return pd.read_csv("chicago_crime_500k.csv") #C:\Dev\AIML\PatrolQ\app\pages\01_Crime_Analysis.py
    except:
            return None

df = load_data()

if df is None:
    st.error("❌ Data file not found")
    st.stop()

st.write(f"**Total Records:** {len(df):,}")

# ====== Crime Type Distribution ======
st.subheader("Crime Type Distribution")

crime_counts = df.groupby('Primary Type')['Primary Type'].agg('count').sort_values(ascending=False).head(15)

fig = px.bar(
    y=crime_counts.index,
    x=crime_counts.values,
    orientation='h',
    title="Top 15 Crime Types in Chicago",
    labels={"x": "Count", "y": "Crime Type"},
    color=crime_counts.values,
    color_continuous_scale="Reds"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("")

# ====== Arrest Statistics ======
st.subheader("Arrest Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    arrest_rate = (df['Arrest'] == True).sum() / len(df) * 100
    st.metric("Overall Arrest Rate", f"{arrest_rate:.1f}%")

with col2:
    domestic_count = (df['Domestic'] == True).sum()
    st.metric("Domestic Incidents", f"{domestic_count:,}")

with col3:
    if (df['Domestic'] == True).sum() > 0:
        domestic_arrest_rate = (
            (df[df['Domestic'] == True]['Arrest'] == True).sum() / 
            (df['Domestic'] == True).sum() * 100
        )
        st.metric("Domestic Arrest Rate", f"{domestic_arrest_rate:.1f}%")

# ====== Arrest Rate by Crime Type ======
st.subheader("Arrest Rate by Crime Type")

arrest_by_type = df.groupby('Primary Type').apply(
    lambda x: (x['Arrest'] == True).sum() / len(x) * 100
).sort_values(ascending=False).head(10)

fig = px.bar(
    x=arrest_by_type.values,
    y=arrest_by_type.index,
    orientation='h',
    title="Top 10 Crime Types by Arrest Rate",
    labels={"x": "Arrest Rate (%)", "y": "Crime Type"},
    color=arrest_by_type.values,
    color_continuous_scale="Greens"
)

st.plotly_chart(fig, use_container_width=True)

# ====== Domestic vs Non-Domestic ======
st.subheader("Domestic vs Non-Domestic Crimes")

col1, col2 = st.columns(2)

with col1:
    domestic_dist = df['Domestic'].value_counts()
    fig_pie = px.pie(
        values=domestic_dist.values,
        names=['Non-Domestic', 'Domestic'],
        title="Domestic vs Non-Domestic",
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    if (df['Domestic'] == True).sum() > 0 and (df['Domestic'] == False).sum() > 0:
        arrest_comparison = pd.DataFrame({
            'Type': ['Domestic', 'Non-Domestic'],
            'Arrest Rate': [
                (df[df['Domestic'] == True]['Arrest'] == True).sum() / 
                (df['Domestic'] == True).sum() * 100,
                (df[df['Domestic'] == False]['Arrest'] == True).sum() / 
                (df['Domestic'] == False).sum() * 100
            ]
        })
        
        fig_arrest = px.bar(
            arrest_comparison,
            x='Type',
            y='Arrest Rate',
            title="Arrest Rate Comparison",
            color='Type',
            color_discrete_sequence=['#4ECDC4', '#FF6B6B']
        )
        st.plotly_chart(fig_arrest, use_container_width=True)

st.success("✅ Crime analysis loaded successfully!")