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
RADIUS_MILES = 1

COMMERCIAL_CATS = ['food_and_drink', 'arts_and_entertainment', 'shopping', 'lifestyle_services', 'lodging']
COMMUNITY_CATS = ['community_and_government', 'healthcare', 'education', 'cultural_and_historic']


def calc_place_counts(retailer_locations: pd.DataFrame, overture_places: pd.DataFrame,
                      retailer_lat='latitude', retailer_long='longitude', place_lat='latitude', place_long='longitude'):

    retailer_locations_df = retailer_locations.copy()
    overture_places_df = overture_places.copy()

    retailer_coords = np.radians(retailer_locations_df[[retailer_lat, retailer_long]].to_numpy())

    overture_places_df['taxonomy_group'] = overture_places_df['taxonomy_hierarchy'].apply(
        lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0
        else None)

    radius_radians = RADIUS_MILES / EARTH_RADIUS_MILES
    category_groups = {
        "commercial_places_within_1_mi": COMMERCIAL_CATS,
        "community_places_within_1_mi": COMMUNITY_CATS
    }

    for feature_name, categories in category_groups.items():

        subset = overture_places_df[overture_places_df['taxonomy_group'].isin(categories)]

        if subset.empty:
            retailer_locations_df[feature_name] = 0
            continue

        place_coords = np.radians(subset[[place_lat, place_long]].to_numpy())

        tree = BallTree(place_coords, metric='haversine')

        retailer_locations_df[feature_name] = tree.query_radius(
            retailer_coords,
            r=radius_radians,
            count_only=True
        )

    return retailer_locations_df

if __name__ == '__main__':

    retailer_locations_df = pd.read_parquet(FINAL_LOCATION_PARQUET_PATH)
    overture_places_df = pd.read_parquet(FILTERED_OVERTURE_PARQUET_PATH)

    retailer_locations_no_nan = retailer_locations_df.dropna(subset=['latitude', 'longitude'])

    place_counts_df = calc_place_counts(retailer_locations_no_nan, overture_places_df)
    
    place_counts_path = Path(OVERTURE_PLACE_COUNTS_PARQUET_PATH)
    place_counts_path.parent.mkdir(parents=True, exist_ok=True)
    place_counts_df.to_parquet(str(place_counts_path), index=False, engine="pyarrow")
    
    print(place_counts_df.head())
