"""

config.py

This module does two things. 
1. It defines constants for file paths used in the project.
2. It loads environment variables from a .env file,
specifically the Socrata app token and ACS API Key required for API access.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

RANDOM_SEED = 42

INTERIM_PARQUET_PATH = "data/interim/lottery_sales_2025"
COMBINED_PARQUET_PATH = "data/interim/combined_lottery_sales_2025.parquet"

LOCATION_PARQUET_PATH = "data/interim/location_data.parquet"
SALES_PARQUET_PATH = "data/interim/sales_data.parquet"

FINAL_LOCATION_PARQUET_PATH = "data/final/location_data.parquet"
FINAL_SALES_PARQUET_PATH = "data/final/sales_data.parquet"

NY_STATE_BLOCK_GROUP_PATH = "data/external/census_shapefile/tl_2025_36_bg.zip"
NY_STATE_BOUNDARIES_PATH = "data/geography/ny_state_boundary.parquet"
NY_STATE_COUNTY_LOOKUP_PATH = "data/geography/ny_state_county_lookup.csv"

OVERTURE_PLACE_COUNTS_PARQUET_PATH = "data/interim/external/merged_with_loc/location_data_overture.parquet"
AADT_GDB_PARQUET_PATH = "data/interim/external/merged_with_loc/data_aadt_2025.parquet"
LODES_WAC_PARQUET_PATH = "data/interim/external/merged_with_loc/location_data_wac.parquet"
LODES_RAC_PARQUET_PATH = "data/interim/external/merged_with_loc/location_data_rac.parquet"
EPA_PARQUET_PATH = "data/interim/external/merged_with_loc/location_data_epa.parquet"
ACS_PARQUET_PATH = "data/interim/external/merged_with_loc/location_data_acs.parquet"

CENSUS_STATE_BOUNDARIES_URL = "https://www2.census.gov/geo/tiger/TIGER2025/STATE/tl_2025_us_state.zip"
OVERTURE_PLACES_AWS_PATH = "s3://overturemaps-us-west-2/release/2026-08-19.0/theme=places/type=place/*"
OVERTURE_REGISTRY_AWS_PATH = "s3://overturemaps-us-west-2/registry/*.parquet"
INTERIM_OVERTURE_PARQUET_PATH = "data/interim/external/overture_data.parquet"
FIRST_SEEN_OVERTURE_PARQUET_PATH = "data/interim/external/overture_first_seen.parquet"
FILTERED_OVERTURE_PARQUET_PATH = "data/interim/external/overture_filtered.parquet"

AADT_CSV_PATH = "data/external/aadt/Annual_Average_Daily_Traffic_AADT_Beginning_1977.csv"
AADT_PARQUET_PATH = "data/interim/external/location_data_aadt.parquet"
AADT_GDB_PATH = "data/external/aadt/GDB_Shapefile_AADT_2024-2025.gdb.zip"

LODES_WAC_CSV_PATH = "data/raw/ny_wac_S000_JT00_2023.csv.gz"
LODES_RAC_CSV_PATH = "data/raw/ny_rac_S000_JT00_2023.csv.gz"

EPA_RAWCSV_PATH = "data/raw/EPA_NY.csv"

ACS_BASE_URL = "https://api.census.gov/data/2024/acs/acs5"
ACS_INTERIM_PARQUET_PATH = "data/interim/external/acs_data_all"

CORRECTIONS_PATH = "data/corrections/address_corrections.csv"

TRAINING_MASTER_PARQUET_PATH = "data/final/master_data.parquet"
TRAIN_DATA_PARQUET_PATH = "data/modeling/train_data.parquet"
VALIDATION_DATA_PARQUET_PATH = "data/modeling/val_data.parquet"
TEST_DATA_PARQUET_PATH = "data/modeling/test_data.parquet"

FLAG_CORRELATION_CSV_PATH = "src/zz_correlation_test/high_correlation_test.csv"
TARGET_CORRELATION_CSV_PATH = "src/zz_correlation_test/target_correlation_test.csv"

TRAIN_PCA_PARQUET_PATH = "data/modeling/pca/X_pca_train.parquet"
VAL_PCA_PARQUET_PATH = "data/modeling/pca/X_pca_val.parquet"
TEST_PCA_PARQUET_PATH = "data/modeling/pca/X_pca_test.parquet"

GMM_PARQUET_DIR = "data/modeling/gmm"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)

SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN")
if not SOCRATA_APP_TOKEN:
    raise RuntimeError("Socrata app token is missing. Please add it to your .env file.")

ACS_API_KEY = os.getenv("ACS_API_KEY")
if not ACS_API_KEY:
    raise RuntimeError("ACS API key is missing. Please add it to your .env file.")
