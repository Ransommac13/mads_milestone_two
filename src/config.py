"""

config.py

This module does two things. 
1. It defines constants for file paths used in the project.
2. It loads environment variables from a .env file,
specifically the Socrata app token required for API access.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

INTERIM_PARQUET_PATH = "data/interim/lottery_sales_2025"
COMBINED_PARQUET_PATH = "data/interim/combined_lottery_sales_2025.parquet"

LOCATION_PARQUET_PATH = "data/interim/location_data.parquet"
SALES_PARQUET_PATH = "data/interim/sales_data.parquet"

CORRECTIONS_PATH = "data/corrections/address_corrections.csv"

FINAL_LOCATION_PARQUET_PATH = "data/final/location_data.parquet"
FINAL_SALES_PARQUET_PATH = "data/final/sales_data.parquet"

CENSUS_STATE_BOUNDARIES_URL = "https://www2.census.gov/geo/tiger/TIGER2025/STATE/tl_2025_us_state.zip"
NY_STATE_BOUNDARIES_PATH = "data/geography/ny_state_boundary.parquet"
OVERTURE_PLACES_AWS_PATH = "s3://overturemaps-us-west-2/release/2026-08-19.0/theme=places/type=place/*"
INTERIM_OVERTURE_PARQUET_PATH = "data/interim/overture_data.parquet"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)

SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN")
if not SOCRATA_APP_TOKEN:
    raise RuntimeError("Socrata app token is missing. Please add it to your .env file.")
