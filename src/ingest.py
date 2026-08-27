"""

ingest.py
"""

from sodapy import Socrata
import pandas as pd
import geopandas as gpd
import censusgeocode as cg

from pathlib import Path

def ingest_lottery_sales_data():
    """
    Ingests lottery sales data from the New York State Open Data API and returns a pandas DataFrame.
    """

    client = Socrata("data.ny.gov",
                    "QlaypJL6gh3tXg9URRakveTqg",
                    username="macleith@umich.edu",
                    password="Monkeybrains13")

    test_results = client.get("xyvi-fbb9", limit=2000)

    return test_results

def sep_sales_and_location_data(api_results):
    """Takes the results from the API and separates the sales data and location data into two separate DataFrames."""

    all_data = pd.DataFrame(api_results)

    location_data = all_data[["agtno", "busnm", "busaddr", "buscity", "buszip", "georeference",
                              ":@computed_region_kjdx_g34t", ":@computed_region_yamh_8v7k", ":@computed_region_wbg7_3whc"]]


    location_data['coordinates'] = location_data['georeference'].str.get('coordinates')
    location_data['longitude'] = location_data['coordinates'].str[0]
    location_data['latitude'] = location_data['coordinates'].str[1]
    location_data = location_data.drop(columns=["georeference", "coordinates"])
    location_data = location_data.drop_duplicates(subset=["agtno"])

    sales_data = all_data.drop(columns=["busnm", "busaddr", "buscity", "buszip", "georeference",
                                        ":@computed_region_kjdx_g34t", ":@computed_region_yamh_8v7k", ":@computed_region_wbg7_3whc"])

    return location_data, sales_data

def geocode_missing_lat_longs(location_df: pd.DataFrame):
    """Geocodes missing latitude and longitude values in the location DataFrame using the Census Geocoder API."""

    missing_locations = location_df[location_df['latitude'].isna() | location_df['longitude'].isna()]

    for index, row in missing_locations.iterrows():
        agtno = str(row['agtno']).strip()
        street = str(row['busaddr']).strip()
        city = str(row['buscity']).strip()
        zip_code = str(row['buszip']).strip()
        try:
            result = cg.address(street, city=city, state='NY', zip_code=zip_code, returntype='locations')
            if result:
                location_df.at[index, 'latitude'] = result[0]['coordinates']['y']
                location_df.at[index, 'longitude'] = result[0]['coordinates']['x']
            else:
                print(f"No geocoding result for agent number {agtno} at address {repr(street)}, {repr(city)}, {repr(zip_code)}")
        except Exception as e:
            print(f"Error geocoding address for agent number {agtno}: {e}")

    return location_df

def convert_to_geodataframe(location_data):
    """Converts the location data DataFrame to a GeoDataFrame."""

    gdf = gpd.GeoDataFrame(
        location_data,
        geometry=gpd.points_from_xy(location_data.longitude, location_data.latitude),
        crs="EPSG:4326"
    )

    return gdf

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('sales_output_file', help="Path to the sales output file (Parquet)")
    parser.add_argument('location_output_file', help="Path to the location output file (Parquet)")

    args = parser.parse_args()

    data = ingest_lottery_sales_data()
    location_data, sales_data = sep_sales_and_location_data(data)

    location_data = geocode_missing_lat_longs(location_data)
    geo_located_data = convert_to_geodataframe(location_data)

    sales_path = Path(args.sales_output_file)
    location_path = Path(args.location_output_file)
    sales_path.parent.mkdir(parents=True, exist_ok=True)
    location_path.parent.mkdir(parents=True, exist_ok=True)

    sales_data.to_parquet(sales_path, index=False, engine="pyarrow")
    geo_located_data.to_parquet(location_path, index=False, engine="pyarrow")
    
    print("Sales data and Location data written to parquet.")
