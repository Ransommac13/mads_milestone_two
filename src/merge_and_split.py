"""

merge_and_split.py

This module merges the lottery retailer sales data with all potential features from external sources
and splits the data into train, validation, and test sets.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import RANDOM_SEED, FINAL_LOCATION_PARQUET_PATH, FINAL_SALES_PARQUET_PATH
from config import TRAINING_MASTER_PARQUET_PATH, TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH, TEST_DATA_PARQUET_PATH

def merge_all_external_features(location_path: Path, sales_path: Path, feature_dir: Path, id_col: str = "agtno"):
    """
    Merge retailer-level parquet files containing potential features from external datasets.
    External datasets include EPA, LODES RAC, LODES WAC, Overture Places, and AADT.

    Assumes:
    - one row per retailer in each file
    - all feature files contain 'agtno'
    - address/location columns are duplicated across feature files
    """

    shared_cols = ['busnm', 'busaddr', 'buscity', 'buszip', 'chain', 'bustype', 'longitude', 'latitude', 'geometry', 'bg_geoid', 'tract_geoid', 'competitors_within_1_mi', 'region']

    location_base = pd.read_parquet(location_path)

    # Confirm location_base has just one row per retailer.
    if location_base[id_col].duplicated().any():
        raise ValueError(f"Duplicate {id_col} values found in location_base.")

    for feature_path in feature_dir.glob("*.parquet"):

        # Don't merge the base file onto itself.
        if feature_path.resolve() == location_path.resolve():
            continue

        features = pd.read_parquet(feature_path)

        if id_col not in features.columns:
            raise ValueError(f"{id_col} not found in {feature_path.name}")

        if features[id_col].duplicated().any():
            raise ValueError(f"Duplicate {id_col} values found in {feature_path.name}")

        # Drop duplicated address columns.
        cols_to_drop = [col for col in shared_cols if col in features.columns]
        features = features.drop(columns=cols_to_drop)

        # Guard against overwriting an existing feature.
        overlapping = set(location_base.columns).intersection(set(features.columns)) - {id_col}
        if overlapping:
            raise ValueError(f"Overlapping columns found in {feature_path.name}: {overlapping}")

        location_base = location_base.merge(features, on=id_col, how="left", validate="one_to_one")

        print(f"Merged {feature_path.name} with location_base. New shape: {location_base.shape}")
    
    sales_data = pd.read_parquet(sales_path)
    sales_data = sales_data[['agtno', 'avg_daily_sales']]
    # Confirm location_base has just one row per retailer.
    if sales_data[id_col].duplicated().any():
        raise ValueError(f"Duplicate {id_col} values found in sales_data.")
    
    master_data_df = sales_data.merge(location_base, on=id_col, how='left', validate='one_to_one')
    # Drops retailers that were removed from the location data due to not physically existing or duplicating another retailer.
    master_data_df = master_data_df.dropna(subset=['bg_geoid', 'tract_geoid'])
    print(f"Merged sales_data with locaiton_base. New shape: {master_data_df.shape}")

    return master_data_df

def clean_master_data(master_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the master data of any columns that are not features or help to identify the retailer."""

    location_cols_to_drop = ['busnm', 'busaddr', 'buscity', 'buszip', 'geometry']
    vestigial_cols_to_drop = ['index_right', 'OBJECTID', 'STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE', 'STATEFP2', 'COUNTYFP2', 'TRACTCE2', 'BLKGRPCE2', 'Rmp']
    irrelevant_cols_to_drop = ['aadt_road_name', 'nonfamily_multiperson_households_proportion']
    redundant_cols_to_drop = ['wac_ce01', 'wac_ce02', 'wac_ce03', 'rac_ce01', 'rac_ce02', 'rac_ce03', 
                              'D1A', 'D1C8_RET', 'D1C8_OFF', 'D1C8_IND', 'D1C8_SVC', 'D1C8_ENT', 'D1C8_ED', 'D1C8_HLTH', 'D1C8_PUB',
                              'households_total', 'age_65_plus_proportion', 'renter_occupied_proportion', 'living_alone_households_proportion']

    cols_to_drop = [*location_cols_to_drop, *vestigial_cols_to_drop, *irrelevant_cols_to_drop, *redundant_cols_to_drop]
    cols_to_drop = [c for c in cols_to_drop if c in master_df.columns]
    
    return master_df.drop(columns=cols_to_drop)

def grouped_train_val_test_split(master_data: pd.DataFrame, group_col: str = "tract_geoid", test_size=0.2, val_size=0.2, random_state=RANDOM_SEED):
    """This function splits the master data into train, validation, and test sets based on a grouping column (e.g., tract_geoid).
    It ensures that all rows with the same group value are in the same split to prevent data leakage."""

    splitter_one = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(splitter_one.split(master_data, groups=master_data[group_col]))

    train_val_data = master_data.iloc[train_val_idx].copy()
    test = master_data.iloc[test_idx].copy()

    relative_val_size = val_size / (1 - test_size)
    splitter_two = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(splitter_two.split(train_val_data, groups=train_val_data[group_col]))

    train = train_val_data.iloc[train_idx].copy()
    val = train_val_data.iloc[val_idx].copy()

    assert set(train["tract_geoid"]).isdisjoint(val["tract_geoid"])
    assert set(train["tract_geoid"]).isdisjoint(test["tract_geoid"])
    assert set(val["tract_geoid"]).isdisjoint(test["tract_geoid"])

    return train, val, test

if __name__ == '__main__':

    location_path = Path(FINAL_LOCATION_PARQUET_PATH)
    sales_path = Path(FINAL_SALES_PARQUET_PATH)
    feature_dir = Path("data/interim/external/merged_with_loc")
    output_path = Path(TRAINING_MASTER_PARQUET_PATH)

    master_data = merge_all_external_features(location_path, sales_path, feature_dir)
    master_data = clean_master_data(master_data)

    master_data.to_parquet(output_path, index=False)
    print(f"Master data saved to {output_path}. Shape: {master_data.shape}")

    train, val, test = grouped_train_val_test_split(master_data, group_col="tract_geoid", test_size=0.2, val_size=0.2)
    train.to_parquet(Path(TRAIN_DATA_PARQUET_PATH), index=False)
    val.to_parquet(Path(VALIDATION_DATA_PARQUET_PATH), index=False)
    test.to_parquet(Path(TEST_DATA_PARQUET_PATH), index=False)
    print("Train, validation, and test sets written to parquet.")
    
    print(f"Train shape: {train.shape}, Validation shape: {val.shape}, Test shape: {test.shape}")
    print("Train data sample:")
    print(train.head())
