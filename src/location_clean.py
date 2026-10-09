"""

location_clean.py

This module contains functions that handle the cleaning and geolocation of location data from the separated location parquet file.
1. It reads the location parquet file containing location data.
2. It geocodes missing latitude and longitude values using the Census Geocoder API.
3. It optionally drops rows with missing latitude or longitude values after attempting to geocode them.
4. It converts the cleaned location data into a GeoDataFrame for spatial analysis.

Input: Location data (Parquet)
Output: Cleaned and geolocated location data (Parquet)

Note: In the future, we may use address_corrections.py to resolve obvious address discrepancies
rather than dropping records the Census Geocoder API cannot geocode.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import censusgeocode as cg
from sklearn.neighbors import BallTree

from config import LOCATION_PARQUET_PATH, FINAL_LOCATION_PARQUET_PATH, NY_STATE_BLOCK_GROUP_PATH

EARTH_RADIUS_MILES = 3958.8

def geocode_missing_lat_longs(location_df: pd.DataFrame):
    """Geocodes missing latitude and longitude values in the location DataFrame using the Census Geocoder API."""

    missing_locations = location_df[location_df['latitude'].isna() | location_df['longitude'].isna()]

    for index, row in missing_locations.iterrows():
        agtno = str(row['agtno']).strip()
        street = str(row['busaddr']).strip()
        city = str(row['buscity']).strip()
        zip_code = str(row['buszip']).strip()
        name = str(row['busnm']).strip()
        try:
            result = cg.address(street, city=city, state='NY', zip_code=zip_code, returntype='locations')
            if result:
                location_df.at[index, 'latitude'] = result[0]['coordinates']['y']
                location_df.at[index, 'longitude'] = result[0]['coordinates']['x']
            else:
                print(f"No geocoding result for agent number {agtno} {name} at address {repr(street)}, {repr(city)}, {repr(zip_code)}")
        except Exception as e:
            print(f"Error geocoding address for agent number {agtno}: {e}")

    unmatched = missing_locations[missing_locations['latitude'].isna() | missing_locations['longitude'].isna()]
    print(f"No geocoding results: {len(unmatched)} retailers")

    return location_df

def drop_locations(locations_df: pd.DataFrame):
    """Drops retailers that upon further inspection either can't be reliably attributed to a Google Maps location or are a duplicate retailer."""

    locations = locations_df.copy()
    
    agtnos_to_drop = {
        '32233' : 'nonexistent', 
        '45277' : 'low confidence',
        '46005' : 'low confidence',
        '75730' : 'low confidence',
        '114708' : 'nonexistent',
        '113146' : 'duplicate retailer',
        '115042' : 'duplicate retailer',
        '115552' : 'duplicate retailer'
        }

    locations["agtno"] = (locations["agtno"].astype("string").str.strip())
    drop_ids = {str(x).strip() for x in agtnos_to_drop.keys()}

    locations = locations[~locations["agtno"].isin(drop_ids)]

    return locations

def convert_to_geodataframe(location_df):
    """Converts the location data DataFrame to a GeoDataFrame."""

    gdf = gpd.GeoDataFrame(
        location_df,
        geometry=gpd.points_from_xy(location_df.longitude, location_df.latitude),
        crs="EPSG:4326"
    )

    return gdf

def attach_census_bg(location_gdf, ny_bg_shapefile_path=NY_STATE_BLOCK_GROUP_PATH):
    """Uses lat-long to assign each retailer a census GEOID, which includes census tract and block group."""

    # Read in the shapefile for NY state census block groups.
    block_groups = gpd.read_file(ny_bg_shapefile_path)
    block_groups = block_groups[['GEOID', 'geometry']].rename(columns={'GEOID' : 'bg_geoid'})

    # Match coordinate reference systems between the ny lottery data and census shapefile.
    location_gdf = location_gdf.to_crs(block_groups.crs)

    # Merge datasets.
    location_gdf = gpd.sjoin(location_gdf, block_groups, how='left', predicate='within')

    # Extract census tract id. Block group id is just census tract plus a 12th digit.
    location_gdf['tract_geoid'] = location_gdf['bg_geoid'].str[:11]

    # Drop column created for the spatial join.
    location_gdf = location_gdf.drop(columns='index_right')

    unmatched=location_gdf[location_gdf['bg_geoid'].isna() & location_gdf.geometry.notna()]
    print(f"No block group geoid: {len(unmatched)} retailers")
    print(f"Total retailers: {len(location_gdf)}")

    return location_gdf

def convert_chain_indicator(location_gdf, chain_col='chain'):

    df = location_gdf.copy()
    df[chain_col] = df[chain_col].astype(int)

    df[chain_col] = np.where(df[chain_col] > 0, 'chain', 'independent')

    return df

def calc_competitor_counts(location_gdf, radius_miles=1, lat_col='latitude', long_col='longitude'):

    retailers_df = location_gdf.copy()

    coords = np.radians(retailers_df[[lat_col, long_col]].to_numpy())

    tree = BallTree(coords, metric="haversine")

    radius_radians = (radius_miles / EARTH_RADIUS_MILES)

    counts = tree.query_radius(coords, r=radius_radians, count_only=True)

    # Each retailer counts itself at distance 0
    retailers_df[f"competitors_within_{radius_miles}_mi"] = counts - 1

    return retailers_df

if __name__ == "__main__":

    location_data = pd.read_parquet(LOCATION_PARQUET_PATH)

    location_data = drop_locations(location_data)
    location_data = geocode_missing_lat_longs(location_data)
    geo_located_data = convert_to_geodataframe(location_data)

    final_location_data = attach_census_bg(geo_located_data, NY_STATE_BLOCK_GROUP_PATH)
    
    final_location_data = convert_chain_indicator(final_location_data)
    final_location_data = calc_competitor_counts(final_location_data, radius_miles=1)

    final_location_data = final_location_data.drop(columns=['correction_reason', 'address_corrected', 'coordinates_corrected', 'geocode_source'])

    final_location_path = Path(FINAL_LOCATION_PARQUET_PATH)
    final_location_path.parent.mkdir(parents=True, exist_ok=True)

    final_location_data.to_parquet(str(final_location_path), index=False, engine="pyarrow")

    print("Final location data written to parquet!")
