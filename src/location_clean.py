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
import geopandas as gpd
import censusgeocode as cg

from config import LOCATION_PARQUET_PATH, FINAL_LOCATION_PARQUET_PATH, NY_STATE_BLOCK_GROUP_PATH

def geocode_missing_lat_longs(location_df: pd.DataFrame):
    """Geocodes missing latitude and longitude values in the location DataFrame using the Census Geocoder API."""

    missing_locations = location_df[location_df['latitude'].isna() | location_df['longitude'].isna()]

    for index, row in missing_locations.iterrows():
        agtno = str(row['agtno']).strip()
        street = str(row['busaddr']).strip()
        city = str(row['buscity']).strip()
        zip_code = str(row['buszip']).strip()
        try:
            result = cg.address(street, city=city, state='NY', zip_code=zip_code, returntype='locations')
            if result:
                location_df.at[index, 'latitude'] = result[0]['coordinates']['y']
                location_df.at[index, 'longitude'] = result[0]['coordinates']['x']
            #else:
            #    print(f"No geocoding result for agent number {agtno} at address {repr(street)}, {repr(city)}, {repr(zip_code)}")
        except Exception as e:
            print(f"Error geocoding address for agent number {agtno}: {e}")

    unmatched = missing_locations[missing_locations['latitude'].isna() | missing_locations['longitude'].isna()]
    print(f"No geocoding results: {len(unmatched)} retailers")

    return location_df

def drop_missing_locations(location_df: pd.DataFrame):
    """Drops rows with missing latitude or longitude values from the location DataFrame.
    Only use after attempting to geocode missing locations.
    Eventually use address_corrections.py to resolve obvious discrepancies."""

    return location_df.dropna(subset=['latitude', 'longitude'])

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

if __name__ == "__main__":

    location_data = pd.read_parquet(LOCATION_PARQUET_PATH)

    location_data = geocode_missing_lat_longs(location_data)
    #location_data = drop_missing_locations(location_data)
    geo_located_data = convert_to_geodataframe(location_data)

    final_location_data = attach_census_bg(geo_located_data, NY_STATE_BLOCK_GROUP_PATH)
    print(final_location_data.sample(5))

    final_location_path = Path(FINAL_LOCATION_PARQUET_PATH)
    final_location_path.parent.mkdir(parents=True, exist_ok=True)

    final_location_data.to_parquet(str(final_location_path), index=False, engine="pyarrow")
