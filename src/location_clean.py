"""

location_clean.py\

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

from config import LOCATION_PARQUET_PATH, FINAL_LOCATION_PARQUET_PATH

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
            else:
                print(f"No geocoding result for agent number {agtno} at address {repr(street)}, {repr(city)}, {repr(zip_code)}")
        except Exception as e:
            print(f"Error geocoding address for agent number {agtno}: {e}")

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

if __name__ == "__main__":

    location_data = pd.read_parquet(LOCATION_PARQUET_PATH)

    location_data = geocode_missing_lat_longs(location_data)
    #location_data = drop_missing_locations(location_data)
    geo_located_data = convert_to_geodataframe(location_data)

    final_location_path = Path(FINAL_LOCATION_PARQUET_PATH)
    final_location_path.parent.mkdir(parents=True, exist_ok=True)

    geo_located_data.to_parquet(str(final_location_path), index=False, engine="pyarrow")
