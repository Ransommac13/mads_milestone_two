"""

address_corrections.py
This module enables users to add address corrections for lottery sales data.
"""

from pathlib import Path

import pandas as pd

from src.config import CORRECTIONS_PATH, LOCATION_PARQUET_PATH

def apply_address_corrections(location_df: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    """Apply approved address corrections by agtno."""

    locations = location_df.copy()

    if corrections.empty:
        locations["address_corrected"] = False
        locations["coordinates_corrected"] = False
        return locations

    locations["agtno"] = (locations["agtno"].astype("string").str.strip())
    corrections["agtno"] = (corrections["agtno"].astype("string").str.strip())
    corrections["longitude"] = pd.to_numeric(corrections["longitude"], errors="coerce")
    corrections["latitude"] = pd.to_numeric(corrections["latitude"], errors="coerce")


    correction_fields = corrections[["agtno", "original_busaddr", "original_buscity", "original_buszip",
                                     "corrected_busaddr", "corrected_buscity", "corrected_buszip",
                                     "correction_reason", "longitude", "latitude", "geocode_source"]].copy()

    correction_fields = correction_fields.rename(columns={
        "longitude": "corrected_longitude",
        "latitude": "corrected_latitude",
        "geocode_source": "correction_geocode_source"
        })

    locations = locations.merge(
        correction_fields,
        on="agtno",
        how="left",
        validate="many_to_one",
    )

    has_address_correction = (locations["corrected_busaddr"].notna() | locations["corrected_buscity"].notna() | locations["corrected_buszip"].notna())
    has_manual_coordinates = (locations["corrected_longitude"].notna() & locations["corrected_latitude"].notna())

    address_mismatch = (has_address_correction & (
        (locations["original_busaddr"].notna() & (locations["busaddr"].astype("string").str.strip() != locations["original_busaddr"].astype("string").str.strip()))
        | (locations["original_buscity"].notna() & (locations["buscity"].astype("string").str.strip() != locations["original_buscity"].astype("string").str.strip()))
        | (locations["original_buszip"].notna() & (locations["buszip"].astype("string").str.strip() != locations["original_buszip"].astype("string").str.strip()))
        ))

    if address_mismatch.any():
        conflicts = locations.loc[address_mismatch,["agtno", "busaddr", "original_busaddr", "buscity", "original_buscity", "buszip", "original_buszip"]]
        print(conflicts)

        raise ValueError(f"Address conflict detected for agtnos: {conflicts['agtno'].tolist()}")
    
    locations.loc[has_address_correction, "busaddr"] = locations.loc[has_address_correction, "corrected_busaddr"]
    locations.loc[has_address_correction, "buscity"] = locations.loc[has_address_correction, "corrected_buscity"]
    locations.loc[has_address_correction, "buszip"] = locations.loc[has_address_correction, "corrected_buszip"]

    locations.loc[has_manual_coordinates, "longitude"] = locations.loc[has_manual_coordinates, "corrected_longitude"]
    locations.loc[has_manual_coordinates, "latitude"] = locations.loc[has_manual_coordinates, "corrected_latitude"]

    locations["address_corrected"] = has_address_correction
    locations["coordinates_corrected"] = has_manual_coordinates

    if "geocode_source" not in locations.columns:
        locations["geocode_source"] = pd.NA
    
    has_source = locations["correction_geocode_source"].notna()
    locations.loc[has_source, "geocode_source"] = locations.loc[has_source, "correction_geocode_source"]

    locations = locations.drop(columns=[
        "original_busaddr", "original_buscity", "original_buszip",
        "corrected_busaddr", "corrected_buscity", "corrected_buszip",
        "corrected_longitude", "corrected_latitude", "correction_geocode_source"])

    return locations

if __name__ == "__main__":

    location_df = pd.read_parquet(LOCATION_PARQUET_PATH)
    corrections_df = pd.read_csv(CORRECTIONS_PATH, dtype={
            "agtno": "string",
            "original_buszip": "string",
            "corrected_buszip": "string"})

    corrected_locations_df = apply_address_corrections(location_df=location_df, corrections=corrections_df)

    corrected_locations_df.to_parquet(LOCATION_PARQUET_PATH, index=False, engine="pyarrow")

    print(f"Corrected addresses or coordinates for {len(corrections_df)} retailers.")
