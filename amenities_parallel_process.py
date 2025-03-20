import pandas as pd
import requests
from geopy.distance import geodesic
import time
from datetime import datetime
import os
from tqdm import tqdm
import json
import concurrent.futures
import numpy as np

def get_batch_start_index(output_file):
    """Determine the starting index by checking existing output file"""
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        return len(existing_df)
    return 0

def save_checkpoint(processed_data, output_file, mode='a'):
    """Save processed batch to CSV"""
    df = pd.DataFrame(processed_data)
    header = not os.path.exists(output_file) if mode == 'a' else True
    df.to_csv(output_file, mode=mode, header=header, index=False)

def process_location(args):
    """Process a single location - designed for parallel execution"""
    row, categories = args
    row_results = {}
    row_results.update(row)  # Keep original data
    
    try:
        for category in categories:
            count, nearest_distance = get_nearby_places(
                row['Latitude'], 
                row['Longitude'], 
                category
            )
            row_results[f'Number_of_{category.capitalize()}_Nearby'] = count
            row_results[f'Distance_to_Nearest_{category.capitalize()}_miles'] = nearest_distance
        
        # Add small random delay to avoid exact simultaneous requests
        time.sleep(0.2 + np.random.random() * 0.3)  # Random delay between 0.2-0.5 seconds
        return row_results
        
    except Exception as e:
        print(f"Error processing row: {str(e)}")
        # Save error information
        row_results['processing_error'] = str(e)
        return row_results

def process_locations_in_batches(input_file, output_file, batch_size=100, max_workers=8):
    # Read input data
    df = pd.read_csv(input_file)
    total_rows = len(df)
    
    # Get starting point from existing output
    start_index = get_batch_start_index(output_file)
    
    if start_index >= total_rows:
        print("All data has already been processed!")
        return
    
    print(f"Starting from index {start_index}")
    
    # Process in batches
    for batch_start in range(start_index, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch = df.iloc[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}: rows {batch_start} to {batch_end}")
        start_time = datetime.now()
        
        # Prepare arguments for parallel processing
        args_list = [(row.to_dict(), categories) for _, row in batch.iterrows()]
        
        results = []
        # Use ThreadPoolExecutor for I/O-bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_location, args) for args in args_list]
            
            # Process results as they complete with progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in worker thread: {str(e)}")
        
        # Save batch results
        save_checkpoint(results, output_file)
        
        # Log batch completion
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Batch completed in {duration}. Saved to {output_file}")
        
        # Save batch metadata for recovery
        metadata = {
            'last_completed_index': batch_end,
            'timestamp': datetime.now().isoformat(),
            'batch_size': batch_size
        }
        with open('batch_metadata.json', 'w') as f:
            json.dump(metadata, f)

def get_nearby_places(lat, lon, category, radius=5000):
    """
    Query OpenStreetMap for nearby places of a specific category.
    
    Args:
        lat (float): Latitude of the location
        lon (float): Longitude of the location
        category (str): Category of place to search for
        radius (int): Search radius in meters (default: 5000)
    
    Returns:
        tuple: (count of places, distance to nearest place in miles)
    """
    # Define category filters for OpenStreetMap
    category_filters = {
        "parks": '["leisure"="park"]',
        "malls": '["shop"="mall"]',
        "metros": '["railway"="station"]["station"="subway"]',
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
    
    # Verify category is valid
    if category not in category_filters:
        raise ValueError(f"Invalid category: {category}")
    
    # Construct Overpass API query
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:25];
    (
        node{category_filters[category]}(around:{radius},{lat},{lon});
        way{category_filters[category]}(around:{radius},{lat},{lon});
        relation{category_filters[category]}(around:{radius},{lat},{lon});
    );
    out body;
    >;
    out skel qt;
    """
    
    try:
        # Make API request with retry mechanism
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.get(overpass_url, params={'data': overpass_query})
                response.raise_for_status()  # Raise exception for bad status codes
                data = response.json()
                break
            except (requests.exceptions.RequestException, ValueError) as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise Exception(f"Failed to fetch data after {max_retries} attempts: {str(e)}")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        
        # Process results
        places = []
        for element in data.get('elements', []):
            if 'tags' in element:
                # Get coordinates based on element type
                if element.get('type') == 'node':
                    place_lat = element.get('lat')
                    place_lon = element.get('lon')
                else:  # way or relation
                    center = element.get('center', {})
                    place_lat = center.get('lat')
                    place_lon = center.get('lon')
                
                if place_lat is not None and place_lon is not None:
                    # Calculate distance in miles
                    distance = geodesic((lat, lon), (place_lat, place_lon)).miles
                    places.append(distance)
        
        # Return results
        if places:
            return len(places), min(places)  # Count and nearest distance
        else:
            return 0, float('inf')  # No places found
            
    except Exception as e:
        print(f"Error querying {category}: {str(e)}")
        return 0, float('inf')  # Return default values on error

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "data/geocode.csv"
    OUTPUT_FILE = "places_analysis_results.csv"
    BATCH_SIZE = 100
    MAX_WORKERS = 8  # Adjust based on your system capabilities and API rate limits
    
    # Categories (from your original code)
    categories = [
        "parks", "malls", "metros", "clubs", "restaurants", "schools", 
        "colleges", "universities", "buses", "trains", "airports", 
        "museums", "libraries", "grocery_stores"
    ]
    
    print(f"Starting parallel batch processing of {INPUT_FILE}")
    print(f"Results will be saved to {OUTPUT_FILE}")
    print(f"Batch size: {BATCH_SIZE}, Max workers: {MAX_WORKERS}")
    
    try:
        process_locations_in_batches(INPUT_FILE, OUTPUT_FILE, BATCH_SIZE, MAX_WORKERS)
        print("\nProcessing completed successfully!")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Progress has been saved.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Progress has been saved and can be resumed from the last successful batch.")