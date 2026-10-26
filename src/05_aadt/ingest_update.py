"""
External dataset: GDB_Shapefile_AADT_2024-2025.gdb.zip
Merge retailer location to the nearest NY AADT road segemnt, to geocode

Merges with the retailer locations by street name and municipality
"""
from re import match

import geopandas as gpd
from pathlib import Path
import numpy as np
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import FINAL_LOCATION_PARQUET_PATH, AADT_GDB_PATH, AADT_GDB_PARQUET_PATH, PROJECT_ROOT
OUTPUT_PATH = PROJECT_ROOT / AADT_GDB_PARQUET_PATH


# keep columns
AADT_KEEP_COL = ["AADT", "RdName", "FC_Text", "Lanes", "Rmp"]
AADT_YEAR = "AADT_2025"
# NY State meters, max distance below this is real meters
#Converts everything into a NY-specific system where units are meters
# gives consistent measurement of how far away is this two points
coordinate_gps = "EPSG:32118"

#retailer farther then this from meter distance, then won't count as a match
max_distance =1600 #1mile

def total_lanes(lanes_col):
    """
    convert the raw value for lanes column
    ex: 2:2 || will be 2+2
    1:1 || will be 1+1
    """
    if pd.isna(lanes_col):
        return np.nan

    numbers = re.findall(r"\d+", lanes_col)
    if not numbers:
        return np.nan
    
    total_road_lanes = sum(int(n) for n in numbers)

    return total_road_lanes

    

def load_road_segemnts(gdb_path, layer):
    
    roads = gpd.read_file(gdb_path, layer=layer)

    available_columns = [col for col in AADT_KEEP_COL if col in roads.columns]
    missing_columns = [col for col in AADT_KEEP_COL if col not in roads.columns]

    if missing_columns:
        print(f"Missing {missing_columns}")

    roads = roads[available_columns + ["geometry"]].copy()
    # remove retired or (highway) rmp


    if "Rmp" in roads.columns:
        regualr_roads = roads["Rmp"].isna() | (roads["Rmp"].astype(str).str.strip() == "")
        roads = roads[regualr_roads] 

    #rename prefix 
    rename_map = {
        "AADT": "aadt_count",
        "RdName": "aadt_road_name",
        "FC_Text": "aadt_functional_class",
        "Lanes": "aadt_lanes",
    }
    roads = roads.rename(columns = {k:v for k, v in rename_map.items() if k in roads.columns})
    roads = roads.drop(columns = [c for c in ("Rmp") if c in roads.columns])

    if "aadt_lanes" in roads.columns:
        roads["aadt_lanes"] = roads["aadt_lanes"].map(total_lanes)

    #convert to meter base coordinate system
    return roads.to_crs(coordinate_gps)

def load_retailer(location_path):

    """
    reads in retailer locations and turn each one's lat/lon into a map point
    """
    path = location_path
    locations = pd.read_parquet(path)

    points = gpd.GeoDataFrame(
        locations,
        #turn each row's coordinates into a map point
        geometry = gpd.points_from_xy(locations["longitude"], locations["latitude"]),
        crs = "EPSG:4326" #plain GPS coordinates
    ) 
    #switch to same meter-based system as the road
    return points.to_crs(coordinate_gps)

def join_nearest(points, roads, max_distance_m):

    """
    attached each retailer to its nearest road segment within max_distance meters
    """

    #can return more than one row per point when two segments
    # every retail stays in result (CHANGE THIS?)
    joined = gpd.sjoin_nearest(
        points, roads, how="left",
        max_distance=max_distance_m, distance_col="dist_to_read_m",
    )

    # keep whichever has the highest traffic count
    if joined["agtno"].duplicated().any():
        joined = (
            joined.sort_values("aadt_count", ascending=False).drop_duplicates(subset="agtno", keep="first")
        )

    return joined

def check_merge(joined):

    total = len(joined)
    matched = joined["aadt_count"].notna().sum()
    unmatched = total - matched 

    return pd.DataFrame([{
            "total_retailers": total, 
            "matched_to_addt": matched,
            "unmatched_no_street_match": unmatched,
        }])
    
if __name__ ==  "__main__":

    # load AADT road traffic data
    roads = load_road_segemnts(AADT_GDB_PATH, AADT_YEAR )
    
    #load retailer locations and turn longitude and latitude into map points
    points = load_retailer(FINAL_LOCATION_PARQUET_PATH)

    # print("Roads CRS:", roads.crs)
    # print("points crs:", points.crs)
    # print("Road bounds:", roads.point_bounds)
    # print("Point bounds:", points.total_bounds)


    #match each retailer to its nearest road within 100 meters and attach traffic count
    enriched_locations = join_nearest(points, roads, max_distance)

    
    validation_results = check_merge(enriched_locations)
    print(validation_results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok = True)
    enriched_locations.drop(columns="geometry").to_parquet(OUTPUT_PATH, index=False)
    print(f"saved: {OUTPUT_PATH}")

