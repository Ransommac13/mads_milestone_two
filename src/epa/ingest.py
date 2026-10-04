"""
ingest.py

Ingest EPA data, perform processing to merge to NY Lottery Retail Sales data and save as parquet file
"""

from heapq import merge
from multiprocessing import Value
from pathlib import Path
import pandas as pd

from src.config import FINAL_LOCATION_PARQUET_PATH, PROJECT_ROOT, EPA_RAWCSV_PATH, EPA_PARQUET_PATH

OUTPUT_PATH = PROJECT_ROOT / EPA_PARQUET_PATH

def load_epa(epa_csv_path):  
    epa_df = pd.read_csv(epa_csv_path)
    return epa_df

def process_epa(epa_df):
    columns = ['OBJECTID','GEOID10','GEOID20','STATEFP','COUNTYFP','TRACTCE','BLKGRPCE','CBSA_POP','CBSA_EMP','HH','D3B', 'NatWalkInd']
    subset_epa_df = epa_df[columns]

    epa_process_df = subset_epa_df[subset_epa_df['STATEFP'] == 36]

    epa_process_df['STATEFP2'] = epa_process_df['STATEFP'].astype(str)
    epa_process_df['COUNTYFP2'] = epa_process_df['COUNTYFP'].astype(str).str.zfill(3)
    epa_process_df['TRACTCE2'] = epa_process_df['TRACTCE'].astype(str).str.zfill(6)
    epa_process_df['BLKGRPCE2'] = epa_process_df['BLKGRPCE'].astype(str)

    epa_process_df['bg_geoid'] = epa_process_df['STATEFP2'] + epa_process_df['COUNTYFP2'] + epa_process_df['TRACTCE2'] + epa_process_df['BLKGRPCE2']
    return epa_process_df

def merge_epa(df1, df2):
    epa_merged_df = pd.merge(df1, df2, on='bg_geoid', how='inner')
    return epa_merged_df

if __name__ == '__main__':

    location_data = pd.read_parquet(FINAL_LOCATION_PARQUET_PATH)
    #location_data = pd.read_parquet("mads_milestone_two/data/final/location_data.parquet")

    #epa_df = load_epa("/Users/brisabol/Documents/MADS/SIADS 696/EPA_SmartLocationDatabase_V3_Jan_2021_Final (2).csv")
    epa_df = load_epa(EPA_RAWCSV_PATH)
    epa_process_df = process_epa(epa_df)
    epa_merge_df = merge_epa(location_data, epa_process_df)

    #epa_merge_df.to_parquet('mads_milestone_two/data/interim/location_data_epa.parquet', index=False)

    #OUTPUT_PATH.parent.mkdir(parents=True, exist_ok = True)
    epa_merge_df.to_parquet(str(OUTPUT_PATH), index=False)
