import pandas as pd
import requests
from geopy.distance import geodesic
import time
#import ace_tools as tools  # For displaying the results

# Function to get nearby locations based on category
def get_nearby_places(lat, lon, category, radius=5000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Define query based on category
    category_filters = {
        "parks": '["leisure"="park"]',
        "malls": '["shop"="mall"]',
        "metros": '["railway"="station"]',
        "clubs": '["amenity"="nightclub"]',
        "restaurants": '["amenity"="restaurant"]',
        "schools": '["amenity"="school"]',
        "colleges": '["amenity"="college"]',
        "universities": '["amenity"="university"]',
        "buses": '["highway"="bus_stop"]',
        "trains": '["railway"="station"]',
        "airports": '["aeroway"="aerodrome"]',
        "museums": '["tourism"="museum"]',
        "libraries": '["amenity"="library"]',
        "grocery_stores": '["shop"="supermarket"]'
    }
    
    if category not in category_filters:
        raise ValueError(f"Invalid category: {category}")

    overpass_query = f"""
    [out:json][timeout:25];
    (
      node{category_filters[category]}(around:{radius},{lat},{lon});
      way{category_filters[category]}(around:{radius},{lat},{lon});
    );
    out center;
    """
    
    try:
        response = requests.get(overpass_url, params={'data': overpass_query})
        data = response.json()
        
        places = []
        for element in data['elements']:
            if element['type'] == 'node':
                place_coords = (element['lat'], element['lon'])
            else:  # way or relation
                place_coords = (element['center']['lat'], element['center']['lon'])
            
            # Calculate distance from the input coordinates in miles
            distance = geodesic((lat, lon), place_coords).miles
            places.append(distance)
            
        if places:
            return len(places), min(places)  # Return count and nearest distance
        return 0, None
        
    except Exception as e:
        print(f"Error for coordinates ({lat}, {lon}) in {category}: {e}")
        return 0, None

# Load the uploaded CSV file
file_path = "geocode.csv"
df = pd.read_csv(file_path).head(10)

# Ensure the CSV file has 'Latitude' and 'Longitude' columns
if "Latitude" not in df.columns or "Longitude" not in df.columns:
    raise ValueError("The CSV file must contain 'Latitude' and 'Longitude' columns.")

# Categories to fetch data for
categories = [
    "parks", "malls", "metros", "clubs", "restaurants", "schools", 
    "colleges", "universities", "buses", "trains", "airports", 
    "museums", "libraries", "grocery_stores"
]

# Initialize columns
for category in categories:
    df[f'Number_of_{category.capitalize()}_Nearby'] = 0
    df[f'Distance_to_Nearest_{category.capitalize()}_miles'] = None

# Process each row
for idx, row in df.iterrows():
    print(f"Processing row {idx + 1}/{len(df)}")
    
    for category in categories:
        count, nearest_distance = get_nearby_places(row['Latitude'], row['Longitude'], category)
        df.at[idx, f'Number_of_{category.capitalize()}_Nearby'] = count
        df.at[idx, f'Distance_to_Nearest_{category.capitalize()}_miles'] = nearest_distance

    # Add a delay to avoid overwhelming the API
    time.sleep(1)

# Save results to CSV
output_file = "places_analysis_results.csv"
df.to_csv(output_file, index=False)

# Display results
#tools.display_dataframe_to_user(name="Places Analysis Results", dataframe=df)