"""

overture_places/ingest.py

This module ingests overture places data.

Note: Currently, this module ingests AUG '26 data, which is inappropriate for 2025 sales data.
Next step is to merge with the GERS registry from Overture Maps and filter by 'first_seen' field.
"""

from pathlib import Path
import bisect
import json
from urllib.request import urlopen

import duckdb
import pandas as pd
import geopandas as gpd

from src.config import CENSUS_STATE_BOUNDARIES_URL, NY_STATE_BOUNDARIES_PATH
from src.config import OVERTURE_PLACES_AWS_PATH, OVERTURE_REGISTRY_AWS_PATH, INTERIM_OVERTURE_PARQUET_PATH, FILTERED_OVERTURE_PARQUET_PATH

def create_duckdb_connection():

    connection = duckdb.connect()

    connection.execute("INSTALL spatial")
    connection.execute("LOAD spatial")

    connection.execute("INSTALL httpfs")
    connection.execute("LOAD httpfs")

    connection.execute("SET s3_region='us-west-2'")

    return connection

def retrieve_ny_state_boundary(output_path: Path, write_to_parquet = True) -> gpd.GeoDataFrame:

    states = gpd.read_file(CENSUS_STATE_BOUNDARIES_URL)

    new_york = states.loc[
        states['STATEFP'] == '36',
        ["STATEFP", "STUSPS", "NAME", "geometry"]
    ].copy()

    if len(new_york) != 1:
        raise ValueError(f"Expected one New York polygon, found {len(new_york)}")

    new_york = new_york.to_crs('OGC:CRS84')

    if write_to_parquet:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_york.to_parquet(output_path, index=False)

    return new_york

def extract_overture_places_data(duckdb_conn: duckdb.DuckDBPyConnection, output_path: Path):

    ny_state_boundaries_path_sql = Path(NY_STATE_BOUNDARIES_PATH).as_posix().replace("'", "''")
    overture_path_sql = OVERTURE_PLACES_AWS_PATH.replace("'", "''")

    ny_place_data = duckdb_conn.sql(f"""
        WITH new_york as (
            SELECT geometry as state_geometry
            FROM read_parquet('{ny_state_boundaries_path_sql}')
            WHERE STATEFP = '36'
        ),
        candidate_places AS (
            SELECT
                id,
                geometry,
                ST_X(geometry) AS longitude,
                ST_Y(geometry) AS latitude,
                basic_category,
                taxonomy.primary AS taxonomy_primary,
                taxonomy.hierarchy AS taxonomy_hierarchy,
                confidence,
                brand.names.primary AS brand_name
            FROM
                read_parquet(
                    '{overture_path_sql}',
                    filename=true,
                    hive_partitioning=true
                )
            WHERE
                bbox.xmax >= -79.8
                AND bbox.xmin <= -71.8
                AND bbox.ymax >= 40.4
                AND bbox.ymin <= 45.1
        )
        SELECT candidate_places.*
        FROM candidate_places
        CROSS JOIN new_york
        WHERE ST_Within(candidate_places.geometry, new_york.state_geometry)
        """
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ny_place_data.write_parquet(str(output_path), compression='zstd', overwrite=True)

    return output_path

def add_first_seen_date(duckdb_conn: duckdb.DuckDBPyConnection, places_input_path: Path, places_output_path: Path):

    places_input_path_sql = places_input_path.as_posix().replace("'", "''")
    overture_registry_path_sql = OVERTURE_REGISTRY_AWS_PATH.replace("'", "''")
    filtered_overture_path_sql = places_output_path.as_posix().replace("'", "''")

    # Identify the local ids within the Overture locations data.
    duckdb_conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE target_ids AS
        SELECT id
        FROM read_parquet('{places_input_path_sql}')
    """)

    # Join Overture locations data with 'first_seen' field from the GERS registry for later filtering.
    duckdb_conn.execute(f"""
        COPY (
            SELECT
                t.id,
                r.first_seen
            FROM target_ids AS t
            INNER JOIN read_parquet(
                '{overture_registry_path_sql}'
            ) AS r
                ON t.id = r.id
        )
        TO '{filtered_overture_path_sql}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
    """)

    print('Written to Parquet!')

    return places_output_path

def merge_places_first_seen(duckdb_conn: duckdb.DuckDBPyConnection, places_input_path: Path, first_seen_path: Path, filtered_places_path: Path):

    places_input_path_sql = places_input_path.as_posix().replace("'", "''")
    first_seen_path_sql = first_seen_path.as_posix().replace("'", "''")
    filtered_places_path_sql = filtered_places_path.as_posix().replace("'", "''")

    duckdb_conn.execute(f"""
        COPY (
            SELECT
                p.*,
                r.first_seen
            FROM read_parquet('{places_input_path_sql}') AS p
            INNER JOIN read_parquet('{first_seen_path_sql}') AS r
                ON p.id = r.id
        )
        TO '{filtered_places_path_sql}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
    """)

    return filtered_places_path

def filter_places_by_date(duckdb_conn: duckdb.DuckDBPyConnection, filtered_places_path: Path, cutoff_date='2025-06-25'):

    filtered_places_path_sql = filtered_places_path.as_posix().replace("'", "''")
    
    duckdb_conn.execute(f"""
        COPY (
            SELECT *
            FROM read_parquet('{filtered_places_path_sql}')
            WHERE CAST(substr(first_seen, 1, 10) AS DATE)
                <= DATE '{cutoff_date}'
        )
        TO '{filtered_places_path_sql}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
    """)

    print(f"Wrote overture places data to {filtered_places_path}, filtered to places that existed on or before {cutoff_date}")

    return filtered_places_path, cutoff_date


def validate_overture_ingest():

    overture_parquet_path_sql = Path(FILTERED_OVERTURE_PARQUET_PATH).as_posix()

    with duckdb.connect() as connection:
        validation = connection.sql(
            f"""
            SELECT
                COUNT(*) AS row_count,
                COUNT(DISTINCT id) AS unique_place_count,
                COUNT(*) FILTER (
                    WHERE geometry IS NULL
                ) AS missing_geometry_count,
                COUNT(*) FILTER (
                    WHERE basic_category IS NULL
                ) AS missing_category_count,
                COUNT(DISTINCT first_seen) AS unique_start_dates
            FROM read_parquet('{overture_parquet_path_sql}')
            """
        ).df()

    return validation

if __name__ == "__main__":

    overture_connection = create_duckdb_connection()
    interim_overture_parquet_path = Path(INTERIM_OVERTURE_PARQUET_PATH)
    first_seen_parquet_path = Path("data/interim/overture_first_seen.parquet")
    filtered_overture_parquet_path = Path(FILTERED_OVERTURE_PARQUET_PATH)

    ## Only uncomment to rerun ingest of New York state boundaries.
    #ny_state_boundaries_path = Path(NY_STATE_BOUNDARIES_PATH)
    #new_york_boundary = retrieve_ny_state_boundary(ny_state_boundaries_path, write_to_parquet=False)

    ## Only uncomment to rerun ingest of overture place data.
    #extract_overture_places_data(overture_connection, interim_overture_parquet_path)

    ## DO NOT uncomment. Matching places to the registry first_seen date takes over an hour.
    #add_first_seen_date(overture_connection, interim_overture_parquet_path, first_seen_parquet_path)

    merge_places_first_seen(overture_connection, interim_overture_parquet_path, first_seen_parquet_path, filtered_overture_parquet_path)
    filter_places_by_date(overture_connection, filtered_overture_parquet_path)
    
    validation_df = validate_overture_ingest()
    print(validation_df.head())

