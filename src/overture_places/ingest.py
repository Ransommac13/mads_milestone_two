"""

overture_places/ingest.py

This module ingests overture places data.

Note: Currently, this module ingests AUG '26 data, which is inappropriate for 2025 sales data.
Next step is to merge with the GERS registry from Overture Maps and filter by 'first_seen' field.
"""

from pathlib import Path

import duckdb
import geopandas as gpd

from src.config import CENSUS_STATE_BOUNDARIES_URL, NY_STATE_BOUNDARIES_PATH, OVERTURE_PLACES_AWS_PATH, INTERIM_OVERTURE_PARQUET_PATH

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

def validate_overture_ingest():

    interim_overture_parquet_path_sql = Path(INTERIM_OVERTURE_PARQUET_PATH).as_posix()

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
                ) AS missing_category_count
            FROM read_parquet('{interim_overture_parquet_path_sql}')
            """
        ).df()

    return validation

if __name__ == "__main__":

    overture_connection = create_duckdb_connection()

    ny_state_boundaries_path = Path(NY_STATE_BOUNDARIES_PATH)
    new_york_boundary = retrieve_ny_state_boundary(ny_state_boundaries_path)

    interim_overture_parquet_path = Path(INTERIM_OVERTURE_PARQUET_PATH)
    overture_parquet_path = extract_overture_places_data(overture_connection, interim_overture_parquet_path)

    validation_df = validate_overture_ingest()
    print(validation_df.head())
