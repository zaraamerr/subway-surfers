import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Load the data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['ARREST_DATE'] = pd.to_datetime(data['ARREST_DATE'])
    return data

# Main App
def main():
    st.title("NYC Crime Data Dashboard")
    st.sidebar.header("Filters")

    # Load your dataset
    file_path = "/home/zara/subway-surfers/data/CLEANED_NYPD_Arrest_Data_YTD" 
    data = load_data(file_path)

    # Sidebar Filters
    boro_options = data['ARREST_BORO'].unique()
    selected_boro = st.sidebar.multiselect("Select Borough(s):", boro_options, default=boro_options)
    selected_offense = st.sidebar.multiselect("Select Offense(s):", data['OFNS_DESC'].unique())

    # Filter the data
    filtered_data = data[data['ARREST_BORO'].isin(selected_boro)]
    if selected_offense:
        filtered_data = filtered_data[filtered_data['OFNS_DESC'].isin(selected_offense)]

    # Visualize Data on a Map
    st.subheader("Crime Map")
    if not filtered_data.empty:
        map_fig = px.scatter_mapbox(
            filtered_data,
            lat="Latitude",  # Ensure you have latitude and longitude columns
            lon="Longitude",
            color="OFNS_DESC",
            hover_data=["ARREST_BORO", "ARREST_DATE"],
            zoom=10,
            mapbox_style="carto-positron"
        )
        st.plotly_chart(map_fig)
    else:
        st.write("No data to display. Adjust your filters.")

    # Heat Map
    st.subheader("Crime Heat Map")
    if not filtered_data.empty:
        # Create a folium map
        heat_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)  # NYC coordinates

        # Prepare the data for the heat map
        heat_data = filtered_data[['Latitude', 'Longitude']].dropna().values.tolist()
        
            # Add HeatMap to the folium map with a custom red gradient
        HeatMap(
            heat_data, 
            radius=15, 
            blur=20, 
            max_zoom=12, 
            gradient={0.2: 'white', 1: 'red'}  # White (low intensity) to Red (high intensity)
        ).add_to(heat_map)

        # Display the map in Streamlit
        st_folium(heat_map, width=700, height=500)
    else:
        st.write("No data to display. Adjust your filters.")


# Run the app
if __name__ == "__main__":
    main()
