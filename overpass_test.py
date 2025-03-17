import requests

def search_schools_overpass(lat, lon, radius=1000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Query for schools
    overpass_query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="school"](around:{radius},{lat},{lon});
      way["amenity"="school"](around:{radius},{lat},{lon});
    );
    out center;
    """
    
    try:
        response = requests.get(overpass_url, params={'data': overpass_query})
        data = response.json()
        
        print(f"Found {len(data['elements'])} schools:")
        for element in data['elements']:
            if 'tags' in element and 'name' in element['tags']:
                print(f"School name: {element['tags']['name']}")
                if element['type'] == 'node':
                    print(f"Location: ({element['lat']}, {element['lon']})")
                else:  # way or relation
                    print(f"Location: ({element['center']['lat']}, {element['center']['lon']})")
                print("---")
                
    except Exception as e:
        print(f"Error: {e}")

# Test with coordinates (example: New York City)
search_schools_overpass(40.7875866,-73.96924) 