import pandas as pd
import geopandas as gpd
import requests

def add_precinct_info_to_df(input_df):
    """
    Add NYPD precinct information to a dataframe containing NYC zip codes.
    
    Parameters:
    -----------
    input_df : pandas DataFrame
        DataFrame containing a 'zip_code' column with NYC ZIP codes
    
    Returns:
    --------
    pandas DataFrame
        Original dataframe with precinct information added
    """
    # Ensure zip_code column exists
    if 'zip_code' not in input_df.columns:
        raise ValueError("Input dataframe must contain a 'zip_code' column")
    
    # Convert zip codes to strings if they aren't already
    input_df['zip_code'] = input_df['zip_code'].astype(str)
    
    # Directly read ZIP code boundaries (assuming a GeoJSON URL)
    print("Downloading NYC ZIP code boundaries...")
    zip_url = "DIRECT_URL_TO_GEOJSON_OR_SHAPEFILE_FOR_ZIP_CODES"
    zip_gdf = gpd.read_file(zip_url)
    
    # Directly read NYPD precinct boundaries (assuming a GeoJSON URL)
    print("Downloading NYPD precinct boundaries...")
    precinct_url = "DIRECT_URL_TO_GEOJSON_OR_SHAPEFILE_FOR_PRECINCTS"
    precinct_gdf = gpd.read_file(precinct_url)
    
    # Ensure both GDFs have the same CRS (Coordinate Reference System)
    precinct_gdf = precinct_gdf.to_crs(zip_gdf.crs)
    
    # Perform spatial join to find which precinct each ZIP code belongs to
    print("Performing spatial join between ZIP codes and precincts...")
    spatial_join = gpd.sjoin(zip_gdf, precinct_gdf, how="left", predicate="intersects")
    
    # Create a lookup dictionary from ZIP code to precinct
    zip_to_precinct = spatial_join.dissolve(by='ZIPCODE', aggfunc='first')
    zip_to_precinct = zip_to_precinct[['precinct']].to_dict()['precinct']
    
    # Add precinct information to the original dataframe
    print("Adding precinct information to the dataframe...")
    input_df['precinct'] = input_df['zip_code'].map(zip_to_precinct)
    
    return input_df

# Example usage
if __name__ == "__main__":
    # Create a sample dataframe with NYC ZIP codes
    sample_df = pd.DataFrame({
        'zip_code': ['10001', '11201', '10028', '11368', '10007'],
        'neighborhood': ['Chelsea/Midtown', 'Brooklyn Heights', 'Upper East Side', 'Corona', 'Financial District']
    })
    
    print("Sample Input DataFrame:")
    print(sample_df)
    
    # Add precinct information
    result_df = add_precinct_info_to_df(sample_df)
    
    print("\nResult DataFrame with Precinct Information:")
    print(result_df)