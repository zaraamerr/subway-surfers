import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.graph_objects as go
import numpy as np

# Load the data
@st.cache_data
def load_datasets():
    # Load crime data
    crime_data = pd.read_csv("data/CLEANED_NYPD_Arrest_Data_YTD")
    crime_data['ARREST_DATE'] = pd.to_datetime(crime_data['ARREST_DATE'])
    
    # Load subway stations data
    subway_stations = pd.read_csv("data/SubwayEntranceData.csv")
    
    return crime_data, subway_stations

@st.cache_data
def load_data_by_date_range(start_date, end_date):
    """
    Load data within a specified date range
    
    Args:
    - start_date: Start of the date range
    - end_date: End of the date range
    
    Returns:
    - Filtered DataFrame 
    """
    # Load full dataset
    crime_data, subway_stations = load_datasets()
    
    # Filter data for the specified date range
    filtered_data = crime_data[
        (crime_data['ARREST_DATE'] >= start_date) & 
        (crime_data['ARREST_DATE'] <= end_date)
    ]
    
    return filtered_data, subway_stations

def calculate_nearby_crime(crime_data, station_lat, station_lon, radius_km=0.5):
    """
    Calculate crimes near a specific station within a given radius
    
    Args:
    - crime_data: DataFrame of crime data
    - station_lat: Latitude of the station
    - station_lon: Longitude of the station
    - radius_km: Radius in kilometers to consider as nearby
    
    Returns:
    - Filtered crime data near the station
    """
    # Haversine formula for distance calculation
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c

    # Calculate distances
    crime_data['distance'] = crime_data.apply(
        lambda row: haversine_distance(station_lat, station_lon, row['Latitude'], row['Longitude']), 
        axis=1
    )
    
    # Filter crimes within radius
    nearby_crimes = crime_data[crime_data['distance'] <= radius_km]
    
    return nearby_crimes

def create_station_specific_map(nearby_crimes, station_lat, station_lon):
    """Create a map focused on a specific station and nearby crimes"""
    station_map = folium.Map(location=[station_lat, station_lon], zoom_start=14)
    
    # Add station marker
    folium.Marker(
        [station_lat, station_lon],
        popup='Station Location',
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(station_map)
    
    # Ensure we have valid lat/lon for crimes
    crime_locations = nearby_crimes[['Latitude', 'Longitude']].dropna()
    
    # Only create heatmap if we have locations
    if not crime_locations.empty:
        heat_data = crime_locations.values.tolist()
        HeatMap(
            heat_data,
            radius=15,
            blur=20,
            max_zoom=1,
            gradient={0.2: 'white', 1: 'red'}
        ).add_to(station_map)
    
    return station_map

def create_crime_visualization(nearby_crimes, viz_type='weekly', start_date=None, end_date=None):
    """
    Create either weekly or daily crime trend visualization
    
    Args:
    - nearby_crimes: DataFrame of nearby crimes
    - viz_type: 'weekly' or 'daily'
    
    Returns:
    - Plotly figure of crime trend
    """
    # Ensure that start_date and end_date are in datetime64 format
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) 

    # Filter the data by the selected date range
    if start_date and end_date:
        nearby_crimes = nearby_crimes[(nearby_crimes['ARREST_DATE'] >= start_date) & (nearby_crimes['ARREST_DATE'] <= end_date)]
    
    if viz_type == 'weekly':
        # Weekly trend using pd.Grouper to group by week start
        weekly_trend = nearby_crimes.groupby(pd.Grouper(key='ARREST_DATE', freq='W-MON')).size()
        
        fig = px.line(
            x=weekly_trend.index, 
            y=weekly_trend.values, 
            title="Weekly Crime Trend",
            labels={'x': 'Week', 'y': 'Number of Crimes'}
        )
        
        # Customize hover template to show week range
        fig.update_traces(
            hovertemplate='Week Starting: %{x}<br>Crimes: %{y}<extra></extra>'
        )
    else:
        # Daily trend (keeping the existing daily implementation)
        daily_trend = nearby_crimes.resample('D', on='ARREST_DATE').size()
        fig = px.line(
            x=daily_trend.index, 
            y=daily_trend.values, 
            title="Daily Crime Trend",
            labels={'x': 'Date', 'y': 'Number of Crimes'}
        )
    
    return fig

def station_crime_analysis(nearby_crimes):
    """Generate comprehensive crime analysis for a station's neighborhood"""
    if nearby_crimes.empty:
        return "No crime data available for this station's neighborhood."
    
    # Get top 3 crimes
    top_crimes = nearby_crimes['OFNS_DESC'].value_counts().head(3)
    
    # Analyze days of the week
    nearby_crimes['DAY'] = nearby_crimes['ARREST_DATE'].dt.day_name()
    peak_days = nearby_crimes['DAY'].value_counts().head(3)
    
    # Build visually appealing analysis
    analysis = """
    ### 🚨 Neighborhood Crime Analysis 🚨

    """
    analysis += "\n **🔍 Top 3 Crimes:**\n"
    for crime, count in top_crimes.items():
        analysis += f"- {crime}: {count}\n"

    analysis += "\n**📅 Most Active Crime Days:**\n"
    for day, count in peak_days.items():
        analysis += f"- {day}: {count}\n"
    
    analysis += f"\n**📊 Total Crimes Recorded:** {len(nearby_crimes)}"
    
    return analysis


# Previous functions from the original dashboard
def create_crime_by_borough_chart(filtered_data):
    crime_by_borough = filtered_data.groupby(['ARREST_BORO', 'OFNS_DESC']).size().reset_index(name='count')
    top_crimes = (crime_by_borough.sort_values('count', ascending=False)
                 .groupby('ARREST_BORO')
                 .head(3)
                 .sort_values(['ARREST_BORO', 'count'], ascending=[True, False]))
    
    fig = px.bar(top_crimes, 
                 x='ARREST_BORO',
                 y='count',
                 color='OFNS_DESC',
                 title='Top 3 Crimes by Borough',
                 labels={'count': 'Number of Arrests', 'ARREST_BORO': 'Borough'},
                 barmode='group')
    return fig

def create_weekly_pattern_chart(filtered_data):
    # Add day of week column
    filtered_data['DAY_OF_WEEK'] = filtered_data['ARREST_DATE'].dt.day_name()
    
    # Define the correct order of days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Calculate arrests by day of week
    weekly_pattern = (filtered_data.groupby('DAY_OF_WEEK')
                     .size()
                     .reindex(days_order)
                     .reset_index(name='count'))
    
    fig = px.line(weekly_pattern, 
                  x='DAY_OF_WEEK',
                  y='count',
                  title='Weekly Crime Pattern',
                  labels={'DAY_OF_WEEK': 'Day of Week', 'count': 'Number of Arrests'})
    
    # Update x-axis to ensure correct day order
    fig.update_xaxes(categoryorder='array', categoryarray=days_order)
    
    return fig

def create_trend_analysis(filtered_data):
    # Calculate daily arrests
    daily_arrests = (filtered_data.groupby('ARREST_DATE')
                    .size()
                    .reset_index(name='count'))
    
    # Calculate rolling average
    daily_arrests['rolling_avg'] = daily_arrests['count'].rolling(window=7).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_arrests['ARREST_DATE'], 
                            y=daily_arrests['count'],
                            name='Daily Arrests',
                            mode='lines',
                            line=dict(color='lightgray')))
    # fig.add_trace(go.Scatter(x=daily_arrests['ARREST_DATE'], 
    #                         y=daily_arrests['rolling_avg'],
    #                         name='7-day Moving Average',
    #                         line=dict(color='red')))
    
    fig.update_layout(title='Arrest Trends Over Time',
                     xaxis_title='Date',
                     yaxis_title='Number of Arrests')
    return fig

def main():
    st.title("NYC Crime Data Dashboard")
    
    # Load datasets
    full_data, subway_stations = load_datasets()

    # Determine default date range
    latest_date = full_data['ARREST_DATE'].max()
    default_start_date = latest_date - pd.Timedelta(days=7)
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["Subway Station Crime Analysis", "City-Wide Crime Analysis"])

     # Date range selector with smart defaults
    st.sidebar.header("Data Range Selection")
    date_range = st.sidebar.slider(
        "Select Date Range",
        min_value=full_data['ARREST_DATE'].min().date(),
        max_value=latest_date.date(),
        value=(default_start_date.date(), latest_date.date())
    )

     # Load data for selected date range
    data, _ = load_data_by_date_range(
        pd.to_datetime(date_range[0]), 
        pd.to_datetime(date_range[1])
    )

    # Extract the selected start and end dates from the date_range tuple
    start_date, end_date = date_range

    st.sidebar.caption("Be aware: Loading the full historical data will take some time!")
    
    with tab1:
        st.sidebar.header("Subway Station Analysis")
        
        # Station Selection
        selected_station = st.sidebar.selectbox(
            "Choose a Subway Station", 
            subway_stations['Constituent Station Name'].unique()
        )
        
        # Get selected station details
        station_info = subway_stations[subway_stations['Constituent Station Name'] == selected_station].iloc[0]
        
        # Calculate nearby crimes
        nearby_crimes = calculate_nearby_crime(
            data, 
            station_info['Latitude'], 
            station_info['Longitude']
        )
        
        # Display station details
        st.subheader(f"Station: {selected_station}")

        # Extract latitude and longitude
        station_lat = station_info['Latitude']
        station_lon = station_info['Longitude']
        
        # Calculate nearby crimes
        radius_km = st.sidebar.slider("Radius (km):", 0.1, 5.0, 0.5, step=0.1)
        nearby_crimes = calculate_nearby_crime(data, station_lat, station_lon, radius_km)
        
        # Display Analysis
        if not nearby_crimes.empty:
            st.subheader(f"Crime Analysis for {selected_station}")
            
            # Show crime statistics
            st.markdown(station_crime_analysis(nearby_crimes))
            
            # Visualize crimes near the station
            st.subheader("Station-Specific Crime Map")
            station_map = create_station_specific_map(nearby_crimes, station_lat, station_lon)
            st_folium(station_map, width=700, height=500)
            
            # Trend Visualization
            st.subheader("Crime Trends Near Station")
            trend_type = st.radio("Select Trend Type:", ["Weekly", "Daily"])
            trend_fig = create_crime_visualization(nearby_crimes, viz_type=trend_type.lower(), start_date=start_date, end_date=end_date)
            st.plotly_chart(trend_fig)
        else:
            st.write("No crimes found near this station within the selected radius.")
        
        
        # Detailed Crime Visualization
        if not nearby_crimes.empty:
            # Crime Type Distribution
            st.subheader("Top 5 Crime Types")
            crime_type_counts = nearby_crimes['OFNS_DESC'].value_counts().head(5)
            fig_pie = px.pie(
                values=crime_type_counts.values, 
                names=crime_type_counts.index, 
                title="Top 5 Crime Types Near Station"
            )
            st.plotly_chart(fig_pie)
    
    with tab2:
        st.sidebar.header("City-Wide Filters")
        
        # Sidebar Filters
        boro_options = data['ARREST_BORO'].unique()
        selected_boro = st.sidebar.multiselect("Select Borough(s):", boro_options, default=boro_options)
        selected_offense = st.sidebar.multiselect("Select Offense(s):", data['OFNS_DESC'].unique())
        
        # Filter the data
        filtered_data = data[data['ARREST_BORO'].isin(selected_boro)]
        if selected_offense:
            filtered_data = filtered_data[filtered_data['OFNS_DESC'].isin(selected_offense)]
        if len(date_range) == 2:
            filtered_data = filtered_data[
                (filtered_data['ARREST_DATE'].dt.date >= date_range[0]) &
                (filtered_data['ARREST_DATE'].dt.date <= date_range[1])
            ]
        
        # Existing visualizations
        st.subheader("Crime Map")
        if not filtered_data.empty:
            map_fig = px.scatter_mapbox(
                filtered_data,
                lat="Latitude",
                lon="Longitude",
                color="OFNS_DESC",
                opacity= 0.5,
                hover_data=["ARREST_BORO", "ARREST_DATE"],
                zoom=10,
                mapbox_style="carto-positron"
            )
            st.plotly_chart(map_fig)
        
        # Heat Map
        # st.subheader("Crime Heat Map")
        # if not filtered_data.empty:
        #     heat_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
        #     heat_data = filtered_data[['Latitude', 'Longitude']].dropna().values.tolist()
        #     HeatMap(
        #         heat_data,
        #         radius=15,
        #         blur=20,
        #         max_zoom=12,
        #         gradient={0.2: 'white', 1: 'red'}
        #     ).add_to(heat_map)
        #     st_folium(heat_map, width=700, height=500)
        
        # Additional visualizations from previous dashboard
        if not filtered_data.empty:
            # Crime by Borough
            st.subheader("Top Crimes by Borough")
            borough_fig = create_crime_by_borough_chart(filtered_data)
            st.plotly_chart(borough_fig)
            
            # Weekly Pattern
            st.subheader("Weekly Crime Pattern")
            weekly_fig = create_weekly_pattern_chart(filtered_data)
            st.plotly_chart(weekly_fig)
            
            # Trend Analysis
            st.subheader("Crime Trends")
            trend_fig = create_trend_analysis(filtered_data)
            st.plotly_chart(trend_fig)

            # Key Statistics
            st.subheader("Key Statistics")
            col1, col2, col3 = st.columns(3)
        
            with col1:
                total_arrests = len(filtered_data)
                st.metric("Total Arrests", f"{total_arrests:,}")
            
            with col2:
                peak_date = (filtered_data.groupby('ARREST_DATE')
                        .size()
                        .idxmax())
                st.metric("Peak Arrest Date", peak_date.strftime('%Y-%m-%d'))
            
            with col3:
                most_common_crime = (filtered_data['OFNS_DESC']
                               .value_counts()
                               .index[0])
            st.metric("Most Common Offense", most_common_crime)

if __name__ == "__main__":
    main()