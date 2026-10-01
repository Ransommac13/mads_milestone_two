"""

location_sales_split.py

This module contains functions that handle the separation of location and sales data from the combined parquet file.
1. It reads the combined parquet file containing all 2025 lottery sales data.
2. It separates the location data and sales data into two separate parquet files.
3. It validates the separation by checking the number of sales records without corresponding location data.

Input: Combined parquet file containing all 2025 lottery sales data.
Output: Two separate parquet files: one for location data and one for sales data.
"""

from pathlib import Path

import duckdb

from config import COMBINED_PARQUET_PATH, LOCATION_PARQUET_PATH, SALES_PARQUET_PATH

def separate_location_and_sales_data(source_parquet_path: Path, location_output_path: Path, sales_output_path: Path):
    """This function separates the location data and sales data from the combined parquet file."""

    location_output_path.parent.mkdir(parents=True, exist_ok=True)
    sales_output_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect() as connection:

        source_relation = connection.read_parquet(str(source_parquet_path), union_by_name=True)
        source_relation.create_view("all_lottery_data", replace=True)

        location_data = connection.sql("""
        SELECT
            agtno,
            busnm,
            busaddr,
            buscity,
            buszip,
            georeference,
            chain,
            bustype
        FROM all_lottery_data
        WHERE agtno IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY agtno
            ORDER BY bus_day DESC
        ) = 1
        """).df()

        sales_data = connection.sql("""
            SELECT * EXCLUDE (
                busnm,
                busaddr,
                buscity,
                buszip,
                georeference,
                chain,
                bustype
            )
            FROM all_lottery_data
            WHERE agtno IS NOT NULL
        """)

        sales_data.write_parquet(str(sales_output_path), compression="zstd")

    location_data['coordinates'] = location_data['georeference'].str.get('coordinates')
    location_data['longitude'] = location_data['coordinates'].str[0]
    location_data['latitude'] = location_data['coordinates'].str[1]
    location_data = location_data.drop(columns=["georeference", "coordinates"])

    location_data.to_parquet(str(location_output_path), index=False,compression="zstd")

    return location_data, sales_output_path

def validate_location_sales_split():
    """This function validates the separation of location and sales data."""

    with duckdb.connect() as connection:
        validation = connection.execute(
            """
            SELECT
                COUNT(*) AS sales_rows,
                COUNT(DISTINCT sales.agtno) AS sales_agents,
                COUNT(DISTINCT locations.agtno) AS location_agents,
                COUNT(*) FILTER (
                    WHERE locations.agtno IS NULL
                ) AS sales_without_location
            FROM read_parquet(?) AS sales
            LEFT JOIN read_parquet(?) AS locations
                USING (agtno)
            """,
            [SALES_PARQUET_PATH, LOCATION_PARQUET_PATH]
        ).df()

    return validation

if __name__ == "__main__":

    combined_parquet_path = Path(COMBINED_PARQUET_PATH)
    location_parquet_path = Path(LOCATION_PARQUET_PATH)
    sales_parquet_path = Path(SALES_PARQUET_PATH)

    location_df, sales_path = separate_location_and_sales_data(combined_parquet_path, location_parquet_path, sales_parquet_path)
    validation_results = validate_location_sales_split()

    if validation_results.loc[0, 'sales_without_location'] == 0:
        if validation_results.loc[0, 'sales_agents'] == validation_results.loc[0, 'location_agents']:
            print('Location and sales data separated successfully')
        else:
            print('Mismatch in number of agents between sales and location data')
    else:
        print(f"Warning: {validation_results.loc[0, 'sales_without_location']} sales records do not have corresponding location data")
