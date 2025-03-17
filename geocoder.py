import pandas as pd
import osmnx as ox
from geopy.distance import geodesic

df_addresses = pd.read_csv('geocode.csv')


def fetch_nearby_places(lat, lon, radius=200000):
    """
    Fetch nearby educational places using multiple OSM tags within a given radius (in meters).
    """
    # Define multiple tags for educational facilities
    tags = {
        'amenity': ['school', 'college', 'university', 'kindergarten'],
        'building': ['school', 'college', 'university'],
    }
    
    try:
        gdf = ox.geometries.geometries_from_point((lat, lon), tags=tags, dist=radius)
        return gdf[['name', 'geometry']]
    except Exception as e:
        print(f"Error fetching educational places: {e}")
        return None


# Add columns to store nearest school and distance
df_addresses['Nearest_School'] = None
df_addresses['School_Distance_km'] = None

found_count = 0
not_found_count = 0

for idx, row in df_addresses.iterrows():
    lat, lon = row['Latitude'], row['Longitude']
    
    # Fetch nearby schools
    schools_gdf = fetch_nearby_places(lat, lon, radius=5000)
    
    if schools_gdf is not None and not schools_gdf.empty:
        min_dist = float('inf')
        nearest_school = None

        for _, school in schools_gdf.iterrows():
            if school['geometry'].geom_type == 'Point':
                school_coords = (school['geometry'].y, school['geometry'].x)
                distance = geodesic((lat, lon), school_coords).km

                if distance < min_dist:
                    min_dist = distance
                    nearest_school = school.get('name', 'Unnamed School')

        df_addresses.at[idx, 'Nearest_School'] = nearest_school
        df_addresses.at[idx, 'School_Distance_km'] = min_dist
        found_count += 1
    else:
        df_addresses.at[idx, 'Nearest_School'] = "Not Found"
        df_addresses.at[idx, 'School_Distance_km'] = None
        not_found_count += 1

total_rows = len(df_addresses)
print(f"\nStatistics:")
print(f"Total addresses processed: {total_rows}")
print(f"Schools found: {found_count}")
print(f"Schools not found: {not_found_count}")
print("\nSample results:")
print(df_addresses.head())