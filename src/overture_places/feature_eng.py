"""

feature_eng.py

This module uses the geometry of lottery retailer locations and places of interest to calculate potential features.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

from src.config import FINAL_LOCATION_PARQUET_PATH, FILTERED_OVERTURE_PARQUET_PATH, OVERTURE_PLACE_COUNTS_PARQUET_PATH

EARTH_RADIUS_MILES = 3958.8
PLACE_CATS = ['place_of_learning', 'gas_station', 'arts_and_entertainment', 'shopping',
              'social_or_community_service', 'restaurant', 'casual_eatery', 'bar']
RADII_LIST = [0.5, 1]

def calc_place_counts(retailer_locations: pd.DataFrame, overture_places: pd.DataFrame, cat_list: list = PLACE_CATS, rad_list: list = RADII_LIST,
                      retailer_lat='latitude', retailer_long='longitude', place_lat='latitude', place_long='longitude'):

    retailer_locations_df = retailer_locations.copy()
    retailer_coords = np.radians(retailer_locations_df[[retailer_lat, retailer_long]].to_numpy())

    for category in cat_list:
        subset = overture_places[overture_places['taxonomy_primary'] == category]
        if subset.empty:
            continue
        
        tree=BallTree(np.radians(subset[[place_lat, place_long]].to_numpy()), metric='haversine')

        for radius in rad_list:
            radius_radians = radius / EARTH_RADIUS_MILES

            retailer_locations_df[f"{category}_within_{radius}_mi"] = tree.query_radius(
                retailer_coords, r=radius_radians, count_only=True 
            )

    return retailer_locations_df

if __name__ == '__main__':

    retailer_locations_df = pd.read_parquet(FINAL_LOCATION_PARQUET_PATH)
    overture_places_df = pd.read_parquet(FILTERED_OVERTURE_PARQUET_PATH)

    retailer_locations_no_nan = retailer_locations_df.dropna(subset=['latitude', 'longitude'])

    place_counts_df = calc_place_counts(retailer_locations_no_nan, overture_places_df, PLACE_CATS, RADII_LIST)
    
    place_counts_path = Path(OVERTURE_PLACE_COUNTS_PARQUET_PATH)
    place_counts_path.parent.mkdir(parents=True, exist_ok=True)
    place_counts_df.to_parquet(str(place_counts_path), index=False, engine="pyarrow")
    
    print(place_counts_df.head())
