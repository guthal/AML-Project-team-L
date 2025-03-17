import streamlit as st
import requests
from geopy.distance import geodesic
from datetime import datetime
import time
import folium
from streamlit_folium import st_folium
import pandas as pd

# Initialize session state variables
if 'search_completed' not in st.session_state:
    st.session_state.search_completed = False
if 'amenities_data' not in st.session_state:
    st.session_state.amenities_data = None
if 'search_coords' not in st.session_state:
    st.session_state.search_coords = None
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None

# Define category filters
CATEGORY_FILTERS = {
    "Parks": ('["leisure"="park"]', "🌳"),
    "Shopping Malls": ('["shop"="mall"]', "🏬"),
    "Metro Stations": ('["railway"="station"]', "🚇"),
    "Nightclubs": ('["amenity"="nightclub"]', "🎉"),
    "Restaurants": ('["amenity"="restaurant"]', "🍽️"),
    "Schools": ('["amenity"="school"]', "🏫"),
    "Colleges": ('["amenity"="college"]', "🎓"),
    "Universities": ('["amenity"="university"]', "🎓"),
    "Bus Stops": ('["highway"="bus_stop"]', "🚌"),
    "Train Stations": ('["railway"="station"]', "🚂"),
    "Airports": ('["aeroway"="aerodrome"]', "✈️"),
    "Museums": ('["tourism"="museum"]', "🏛️"),
    "Libraries": ('["amenity"="library"]', "📚"),
    "Grocery Stores": ('["shop"="supermarket"]', "🛒")
}

def get_nearby_amenities(lat, lon, category_filter, radius=5000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    overpass_query = f"""
    [out:json][timeout:25];
    (
        node{category_filter}(around:{radius},{lat},{lon});
        way{category_filter}(around:{radius},{lat},{lon});
        relation{category_filter}(around:{radius},{lat},{lon});
    );
    out body;
    >;
    out skel qt;
    """
    
    try:
        response = requests.get(overpass_url, params={'data': overpass_query})
        data = response.json()
        
        amenities = []
        for element in data['elements']:
            if 'tags' in element:
                amenity_info = {
                    'name': element['tags'].get('name', 'Unnamed'),
                    'type': element['tags'].get('amenity') or 
                           element['tags'].get('leisure') or 
                           element['tags'].get('shop') or 
                           element['tags'].get('railway') or 
                           element['tags'].get('tourism') or 
                           element['tags'].get('highway') or 
                           'Unknown',
                    'coordinates': (
                        element.get('lat', element.get('center', {}).get('lat')),
                        element.get('lon', element.get('center', {}).get('lon'))
                    )
                }
                
                if amenity_info['coordinates'][0] is not None:
                    distance = geodesic((lat, lon), amenity_info['coordinates']).kilometers
                    amenity_info['distance'] = round(distance, 2)
                    amenities.append(amenity_info)
        
        return sorted(amenities, key=lambda x: x['distance'])
        
    except Exception as e:
        st.error(f"Error fetching amenities: {str(e)}")
        return []

def create_map(lat, lon, amenities, category_name):
    try:
        m = folium.Map(location=[lat, lon], zoom_start=13)
        
        # Add marker for search location
        folium.Marker(
            [lat, lon],
            popup="Search Location",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
        
        # Get emoji for category
        emoji = CATEGORY_FILTERS[category_name][1]
        
        # Add markers for each amenity
        for amenity in amenities:
            folium.Marker(
                [amenity['coordinates'][0], amenity['coordinates'][1]],
                popup=f"{emoji} {amenity['name']}<br>Distance: {amenity['distance']} km",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)
        
        return m
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Nearby Amenities Finder", layout="wide")
    
    st.title("🗺️ Nearby Amenities Finder")
    st.write("Find various amenities near your location!")
    
    # Create columns for input
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        lat = st.number_input("Latitude", value=51.5074, format="%.6f")
    with col2:
        lon = st.number_input("Longitude", value=-0.1278, format="%.6f")
    with col3:
        selected_category = st.selectbox(
            "Select Category",
            options=list(CATEGORY_FILTERS.keys()),
            format_func=lambda x: f"{CATEGORY_FILTERS[x][1]} {x}"
        )
    
    if st.button("Search Amenities") or st.session_state.search_completed:
        try:
            # Only perform search if parameters changed or first search
            current_params = (lat, lon, selected_category)
            if (not st.session_state.search_completed or 
                current_params != st.session_state.search_coords):
                
                with st.spinner(f"Searching for {selected_category.lower()}..."):
                    start_time = datetime.now()
                    amenities = get_nearby_amenities(
                        lat, lon, 
                        CATEGORY_FILTERS[selected_category][0]
                    )
                    end_time = datetime.now()
                    
                    # Store results in session state
                    st.session_state.amenities_data = amenities
                    st.session_state.search_coords = current_params
                    st.session_state.processing_time = (end_time - start_time).total_seconds()
                    st.session_state.search_completed = True
            
            # Use stored results
            amenities = st.session_state.amenities_data
            
            if not amenities:
                st.error(f"No {selected_category.lower()} found in the area!")
                return
            
            # Display processing time
            st.info(f"Processing time: {st.session_state.processing_time:.2f} seconds")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Map", "Table", "Summary"])
            
            with tab1:
                m = create_map(lat, lon, amenities, selected_category)
                if m:
                    st_folium(m, width=800, height=500)
            
            with tab2:
                df = pd.DataFrame(amenities)
                if not df.empty:
                    df = df[['name', 'type', 'distance']]
                    df.columns = ['Name', 'Type', 'Distance (km)']
                    st.dataframe(df, use_container_width=True)
            
            with tab3:
                if amenities:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"Total {selected_category}", len(amenities))
                    with col2:
                        st.metric("Nearest Distance", f"{amenities[0]['distance']:.2f} km")
                    with col3:
                        avg_distance = sum(a['distance'] for a in amenities)/len(amenities)
                        st.metric("Average Distance", f"{avg_distance:.2f} km")
                    
                    st.write("### Search Details")
                    st.write(f"- Search Location: ({lat}, {lon})")
                    st.write("- Search Radius: 5 kilometers")
                    st.write("- Data Source: OpenStreetMap via Overpass API")
            
            # Add download button for CSV
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Data as CSV",
                    data=csv,
                    file_name=f"{selected_category.lower()}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Add clear results button
    if st.session_state.search_completed:
        if st.button("Clear Results"):
            st.session_state.search_completed = False
            st.session_state.amenities_data = None
            st.session_state.search_coords = None
            st.session_state.processing_time = None
            st.experimental_rerun()

if __name__ == "__main__":
    main() 