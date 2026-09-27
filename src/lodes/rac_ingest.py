"""
This file will ingests NY LODES Residence Area Characteristics (RAC) data and merges it to 
retailer locations by census block group
"""

from heapq import merge
from multiprocessing import Value
from pathlib import Path
import pandas as pd

from src.config import FINAL_LOCATION_PARQUET_PATH, LODES_RAC_CSV_PATH, LODES_RAC_PARQUET_PATH, PROJECT_ROOT

OUTPUT_PATH = PROJECT_ROOT / LODES_RAC_PARQUET_PATH

RAC_FEATURE_COLUMNS = [
    "C000", # total jobs held by residents of this block
    "CE01", # jobs with earnings $1250/month or less
    "CE02", # jobs with earnings $1251-$3333/month
    "CE03", # jobs with earnings greater than  $3333
]

def load_lodes_rac(rac_csv_path):
    
    rac_df = pd.read_csv(rac_csv_path, dtype={"h_geocode": "string"}, compression="gzip")
    rac_df['bg_geoid'] = rac_df["h_geocode"].str[:12]

    return rac_df

def aggregate_rac_to_block_group(rac_df):
    
    # check if expected column actually exists in the file
    missing_cols = [col for col in RAC_FEATURE_COLUMNS if col not in rac_df.columns]
    if missing_cols:
        raise ValueError(f"Expected LODES columns not found in file: {missing_cols}")

    # group all blocks sharing the same block group
    bg_rac = (
        rac_df.groupby("bg_geoid")[RAC_FEATURE_COLUMNS]
        .sum()
        .reset_index()
    )

        
    # rename -> rac_c000 so these columns are clearly labeled as WAC feature once merged
    bg_rac = bg_rac.rename(columns= {col: f"rac_{col.lower()}" for col in RAC_FEATURE_COLUMNS})

    return bg_rac

def merge_rac_to_locations(location_df, bg_rac_df):

    locations = location_df.copy()
    locations["bg_geoid"] = locations["bg_geoid"].astype("string")
    bg_rac_df = bg_rac_df.copy()
    bg_rac_df["bg_geoid"] = bg_rac_df["bg_geoid"].astype("string")

    merged = locations.merge(bg_rac_df, on="bg_geoid", how="left", validate="many_to_one")

    return merged

def check_merge(merged_df):

    """
    Print out the matching reult 

    """

    total = len(merged_df)
    matched = merged_df["rac_c000"].notna().sum()
    unmatched = total - matched

    return pd.DataFrame([{
        "total_retailers": total, 
        "matched_to_rac": matched,
        "unmatched": unmatched,
    }])


if __name__ ==  "__main__":

    
    location_data = pd.read_parquet(FINAL_LOCATION_PARQUET_PATH)

    rca_data = load_lodes_rac(LODES_RAC_CSV_PATH)
    bg_rac_data = aggregate_rac_to_block_group(rca_data)

    enriched_locations = merge_rac_to_locations(location_data, bg_rac_data)
    validation_results = check_merge(enriched_locations)
    print(validation_results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok = True)
    enriched_locations.to_parquet(str(OUTPUT_PATH), index=False)
