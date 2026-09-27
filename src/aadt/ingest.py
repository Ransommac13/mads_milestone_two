"""
External dataset: NYS Annual Average Daily Traffic Dataset
AADT records each row identifies a road segment by Road Name, Muicipality, County, and where 
segment begins and ends. 

Merges with the retailer locations by street name and municipality
"""
from genericpath import exists
import re 
from pathlib import Path

import pandas as pd

from src.config import FINAL_LOCATION_PARQUET_PATH, AADT_CSV_PATH, AADT_PARQUET_PATH, PROJECT_ROOT
OUTPUT_PATH = PROJECT_ROOT / AADT_PARQUET_PATH

# to normalize abbreviation between the two datasets
STREET_SUFFIX_MAP = {
    "STREET": "ST",
    "AVENUE": "AVE",
    "BOULEVARD": "BLVD",
    "DRIVE": "DR",
    "CIRCLE": "CIR",
    "LANE": "LN",
    "ROAD": "RD",
    "PLACE": "PL",
    "PARKWAY": "PKWY",
    "TERRACE": "TER",
    "HIGHWAY": "HWY",
    "COURT": "CT",
    "TURNPIKE": "TPKE"
}

def normalize_streetname(raw_name):

    """
    standardize common suffix abbreviations for street name
    """

    # treat missing street name as empty string
    if pd.isna(raw_name):
        return " "
    name = str(raw_name).strip().upper()
    
    # drop punctuation
    name = re.sub(r"[^\w\s]", "", name)

    # split the name into individual words
    words = name.split()
    # replace any word found in the suffix map
    words = [STREET_SUFFIX_MAP.get(word, word) for word in words]

    return "".join(words)

def extract_streetname_from_adress(busaddr):
    """
    remove leading house numbers from a bussiness address
    """

    # missing street name will be replaced as empty string
    if pd.isna(busaddr):
        return " "

    # drop leading house / unit number / dashes
    stripped = re.sub(r"^\s*\d+[\w\-]*\s+", "", str(busaddr).strip())

    # return just the street name
    return normalize_streetname(stripped)

def load_aadt_data(aadt_csv_path):
    """
    load in AADT csv file and normalize its street name and municipality fields

    """

    aadt_df = pd.read_csv(aadt_csv_path, thousands= ",")
    aadt_df["Count"] = pd.to_numeric(aadt_df["Count"], errors="coerce").astype("Int64")
    # normalize road name for joining
    aadt_df["road_name_norm"] = aadt_df["Road Name"].apply(normalize_streetname)
    # normalized municipality name for joining
    aadt_df["municipality_norm"] = aadt_df["Municipality"].str.strip().str.upper()

    return aadt_df

def aggregate_aadt_by_street(aadt_df):
    """

    """
    # Group all AADT segment by street + municipality
    street_aadt = (
        aadt_df.groupby(["road_name_norm", "municipality_norm"])["Count"]
        .agg(aadt_mean="mean", addt_max = "max", addt_segment_count = "count")
        .reset_index()
    )
    return street_aadt

def merge_addt_to_locations(location_df, street_aadt_df):
    
    """
    Merge street-level AADT features onto retailer locations by normalized street name + municipality

    """

    locations = location_df.copy()
    # pull normalized street name from each retailer's address
    locations["road_name_norm"] = locations["busaddr"].apply(extract_streetname_from_adress)
    # normalize each retailer's city to match AADT's municipality
    locations["municipality_norm"] = locations["buscity"].str.strip().str.upper()

    # join key: normalized street name + municipality 
    merged = locations.merge(
        street_aadt_df,
        on=["road_name_norm", "municipality_norm"],
        how = "left"

    )

    return merged

def validate_aadt_merge(merged_df):

    """
    Print out the matching result
    
    """
    
    total = len(merged_df) 
    # rows where an AADT match was found
    matched = merged_df["aadt_mean"].notna().sum()
    unmatched = total - matched

    return pd.DataFrame([{
        "total_retailers": total, 
        "matched_to_addt": matched,
        "unmatched_no_street_match": unmatched,
    }])

if __name__ ==  "__main__":

    #load cleaned retailer location
    location_data = pd.read_parquet(FINAL_LOCATION_PARQUET_PATH)

    # load and normalize the raw AADT data
    aadt_data = load_aadt_data(AADT_CSV_PATH)
    # collapse AADT to one row per street + municipality 
    street_aadt_data = aggregate_aadt_by_street(aadt_data)
    # Attach AADT features to each retailer 
    enriched_locations = merge_addt_to_locations(location_data, street_aadt_data)
    validation_results = validate_aadt_merge(enriched_locations)
    print(validation_results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok = True)
    enriched_locations.to_parquet(str(OUTPUT_PATH), index=False)




