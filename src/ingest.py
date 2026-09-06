"""

ingest.py

This module containts functions that handle the ingestion of 2025 lottery sales data using the Socrata API.
1. It creates a Socrata client using the app token from the .env file.
2. It ingests 2025 lottery sales data in pages, saving each page as a parquet file in the specified output directory.
3. It validates the ingested parquet pages by checking the row count and date range.
4. It combines all the parquet pages into a single parquet file for easier analysis and storage.

Input: None
Output: Combined parquet file containing all 2025 lottery sales data.

Note: Due to the large file size of the combined parquet file,
it is not stored in the GitHub repository. Instead, it is generated locally by running the ingest.py script.
The combined parquet file is used as input in location_sales_split.py.
"""

from pathlib import Path

from sodapy import Socrata
from dotenv import load_dotenv
import pandas as pd
import duckdb

from config import SOCRATA_APP_TOKEN, INTERIM_PARQUET_PATH, COMBINED_PARQUET_PATH

load_dotenv()

def create_Socrata_client():
    """This function creates a Socrata client using the app token from the .env file."""

    return Socrata(
        "data.ny.gov",
        SOCRATA_APP_TOKEN,
        timeout=120
    )

def ingest_2025_sales_data(
    client: Socrata,
    output_directory: Path,
    page_size: int = 50000
):
    """This function ingests 2025 lottery sales data in pages,
    saving each page as a parquet file in the specified output directory."""

    output_directory.mkdir(parents=True, exist_ok=True)

    where_clause = (
        "bus_day >= '2025-01-01T00:00:00.000' "
        "AND bus_day < '2026-01-01T00:00:00.000'"
    )

    offset = 0
    page_number = 0
    total_downloaded = 0

    while True:

        records = client.get(
            "xyvi-fbb9",
            where=where_clause,
            limit=page_size,
            offset=offset,
            order="bus_day ASC, agtno ASC"
        )

        if not records:
            break

        page_df = pd.DataFrame.from_records(records)

        output_path = (
            output_directory
            / f"lottery_2025_part_{page_number:04d}.parquet"
        )

        page_df.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")

        rows_received = len(page_df)
        total_downloaded += rows_received
        if rows_received < page_size:
            break

        offset += rows_received
        page_number += 1

    return total_downloaded

def validate_parquet_pages():
    """This function validates the ingested parquet pages
    by checking the row count and date range."""

    validation_results = duckdb.sql("""
        SELECT
            COUNT(*) AS row_count,
            MIN(bus_day) AS earliest_date,
            MAX(bus_day) AS latest_date
        FROM
            read_parquet(
            'data/interim/lottery_sales_2025/*.parquet',
            union_by_name=true
        )
    """).df()

    return validation_results

def combine_parquet_pages():
    """This function combines all the parquet pages into a single parquet file for easier analysis and storage."""

    combined_parquet_path = Path(COMBINED_PARQUET_PATH)
    combined_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_pages_dir = Path(INTERIM_PARQUET_PATH)
    parquet_input_pattern = str(parquet_pages_dir / "*.parquet")

    with duckdb.connect() as connection:
        combined_parquets = connection.read_parquet(parquet_input_pattern, union_by_name=True)
        combined_parquets.write_parquet(COMBINED_PARQUET_PATH, compression="zstd")

if __name__ == "__main__":

    #client = create_Socrata_client()
    #output_directory = Path(INTERIM_PARQUET_PATH)
    #total_records_downloaded = ingest_2025_sales_data(client=client, output_directory=output_directory, page_size=50000)
    combine_parquet_pages()
    print('parquet pages combined successfully')
