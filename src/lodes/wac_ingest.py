"""
This file will ingests NY LODES Workplace Area Characteristics (WAC) data and merges it to 
retailer locations by census block group
"""

from heapq import merge
from multiprocessing import Value
from pathlib import Path
import pandas as pd

from src.config import FINAL_LOCATION_PARQUET_PATH, LODES_WAC_CSV_PATH, LODES_WAC_PARQUET_PATH, PROJECT_ROOT

OUTPUT_PATH = PROJECT_ROOT / LODES_WAC_PARQUET_PATH

# Job-count columns to keep from LODES WAC. C000
# CNS -> jobs by industry sector 
# CE -> jobs by earnings tier

"""
CNS01  (Agriculture, Forestry, Fishing and Hunting)
CNS02 (Mining, Quarrying, and Oil and Gas Extraction)
CNS03  (Utilities)
CNS04  (Construction)
CNS05  (Manufacturing)
CNS06  (Wholesale Trade)
CNS07  (Retail Trade)
CNS08 (Transportation and Warehousing)
CNS09 (Information)
CNS10 (Finance and Insurance)
CNS11 (Real Estate and Rental and Leasing)
CNS12  (Professional, Scientific, and Technical Services)
CNS13  (Management of Companies and Enterprises)
CNS14  (Administrative and Support and Waste
Management and Remediation Services)
CNS15  (Educational Services)
CNS16 (Health Care and Social Assistance)
CNS17  (Arts, Entertainment, and Recreation)
CNS18  (Accommodation and Food Services)
CNS19  (Other Services [except Public Administration])
CNS20  (Public Administration)


"""

CNS_COLUMNS = ["CNS{:02d}".format(i) for i in range (1,21)]

WAC_FEATURE_COLUMNS = CNS_COLUMNS + [
    "C000", # total jobs
    "CE01", # jobs with earnings $1250/month or less
    "CE02", # jobs with earnings $1251-$3333/month
    "CE03", # jobs with earnings greater than  $3333
]

def load_lodes_wac(wac_csv_path):
    # read the gzipped CSV 
    wac_df = pd.read_csv(wac_csv_path, dtype={"w_geocode": "string"}, compression="gzip")

    # w_geocode is a 15-digit census block code, the first 12 digits are the block group
    wac_df['bg_geoid'] = wac_df["w_geocode"].str[:12]

    return wac_df

def aggregate_wac_to_block_group(wac_df):

    # check if expected column actually exists in the file
    missing_cols = [col for col in WAC_FEATURE_COLUMNS if col not in wac_df.columns]
    if missing_cols:
        raise ValueError(f"Expected LODES columns not found in file: {missing_cols}")

    # group all blocks sharing the same block group
    bg_wac = (
        wac_df.groupby("bg_geoid")[WAC_FEATURE_COLUMNS]
        .sum()
        .reset_index()
    )
    # rename -> wac_c000 so these columns are clearly labeled as WAC feature once merged
    bg_wac = bg_wac.rename(columns= {col: f"wac_{col.lower()}" for col in WAC_FEATURE_COLUMNS})

    return add_proportion(bg_wac)

JOB_COL = ['CE01', 'CE02']

def add_proportion(bg_wac):

    total_jobs = bg_wac["wac_c000"].where(bg_wac["wac_c000"] > 0)

    for col in JOB_COL:
        name = f"wac_{col.lower()}"
        bg_wac[f"{name}_prop"] = bg_wac[name] / total_jobs

    return bg_wac


def merge_wac_to_locations(location_df, bg_wac_df):

    locations = location_df.copy()
    locations["bg_geoid"] = locations["bg_geoid"].astype("string")
    bg_wac_df = bg_wac_df.copy()
    bg_wac_df["bg_geoid"] = bg_wac_df["bg_geoid"].astype("string")

    merged = locations.merge(bg_wac_df, on="bg_geoid", how="left", validate="many_to_one")

    return merged

def check_merge(merged_df):

    total = len(merged_df)
    print(enriched_locations.columns.tolist())
    matched = merged_df["wac_c000"].notna().sum()
    unmatched = total - matched

    return pd.DataFrame([{
        "total_retailers": total, 
        "matched_to_wac": matched,
        "unmatched": unmatched,
        "pct_matched": round(100 * matched / total, 2) if total else 0,
    }])


if __name__ ==  "__main__":
    location_data = pd.read_parquet(FINAL_LOCATION_PARQUET_PATH)

    wca_data = load_lodes_wac(LODES_WAC_CSV_PATH)
    bg_wac_data = aggregate_wac_to_block_group(wca_data)

    enriched_locations = merge_wac_to_locations(location_data, bg_wac_data)
    validation_results = check_merge(enriched_locations)
    print(validation_results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok = True)
    enriched_locations.to_parquet(str(OUTPUT_PATH), index=False)
