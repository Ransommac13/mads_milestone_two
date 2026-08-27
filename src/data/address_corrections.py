"""

address_corrections.py
This module enables users to add address corrections for lottery sales data.
"""

import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

from src.config import CORRECTIONS_PATH

def load_address_corrections() -> pd.DataFrame:
    """Load the address-correction ledger."""

    CORRECTION_COLUMNS = [
        "agtno",
        "original_busaddr", "original_buscity", "original_buszip",
        "corrected_busaddr", "corrected_buscity", "corrected_buszip", "correction_reason",
        "source", "added_at"
        ]

    corrections_file_path = Path(CORRECTIONS_PATH)
    if not corrections_file_path.exists():
        return pd.DataFrame(columns=CORRECTION_COLUMNS)

    return pd.read_csv(
        CORRECTIONS_PATH,
        dtype={
            "agtno": "string",
            "original_buszip": "string",
            "corrected_buszip": "string",
        }
    )

def add_address_correction(agtno, original_busaddr, original_buscity, original_buszip,
                           corrected_busaddr, corrected_buscity, corrected_buszip, correction_reason, source):
    """Add a new address correction to the ledger."""

    corrections_df = load_address_corrections()
    agtno = str(agtno).strip()

    already_exists = (corrections_df["agtno"].astype("string") == agtno).any()

    if already_exists:
        raise ValueError(f"A correction already exists for agent number {agtno}.")

    new_correction = pd.DataFrame([{
        "agtno": agtno,
        "original_busaddr": original_busaddr,
        "original_buscity": original_buscity,
        "original_buszip": str(original_buszip).strip(),
        "corrected_busaddr": corrected_busaddr,
        "corrected_buscity": corrected_buscity,
        "corrected_buszip": str(corrected_buszip).strip(),
        "correction_reason": correction_reason,
        "source": source,
        "added_at": datetime.now(timezone.utc).isoformat(),
    }])

    corrections_df = pd.concat(
        [corrections_df, new_correction],
        ignore_index=True,
    )

    Path(CORRECTIONS_PATH).parent.mkdir(parents=True,exist_ok=True)

    corrections_df.to_csv(CORRECTIONS_PATH, index=False)

def apply_address_corrections(location_df: pd.DataFrame) -> pd.DataFrame:
    """Apply approved address corrections by agtno."""

    locations = location_df.copy()
    corrections = load_address_corrections()

    if corrections.empty:
        locations["address_corrected"] = False
        return locations

    locations["agtno"] = (locations["agtno"].astype("string").str.strip())
    corrections["agtno"] = (corrections["agtno"].astype("string").str.strip())

    correction_fields = corrections[[
        "agtno",
        "original_busaddr",
        "corrected_busaddr",
        "corrected_buscity",
        "corrected_buszip",
        "correction_reason",
        "source",
    ]]

    locations = locations.merge(
        correction_fields,
        on="agtno",
        how="left",
        validate="many_to_one",
    )

    has_correction = locations["corrected_busaddr"].notna()

    address_mismatch = has_correction & (locations["busaddr"] != locations["original_busaddr"])
    if address_mismatch.any():
        conflicts = locations.loc[address_mismatch, ["agtno", "busaddr", "original_busaddr"]]
        for _, row in conflicts.iterrows():
            print(f"Conflict for agent number {row['agtno']}: correction orig address '{row['original_busaddr']}' differs from current address '{row['busaddr']}'")

        raise ValueError(f"Address conflict detected for agent number {conflicts['agtno'].tolist()}")

    locations["address_corrected"] = has_correction

    locations.loc[has_correction, "busaddr"] = (
        locations.loc[has_correction, "corrected_busaddr"]
    )

    locations.loc[has_correction, "buscity"] = (
        locations.loc[has_correction, "corrected_buscity"]
    )

    locations.loc[has_correction, "buszip"] = (
        locations.loc[has_correction, "corrected_buszip"]
    )

    return locations

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--agtno", required=True, type=str, help="Agent number to apply corrections to")
    parser.add_argument("--o_addr", required=True, type=str, help="Original business address")
    parser.add_argument("--o_city", required=True, type=str, help="Original business city")
    parser.add_argument("--o_zip", required=True, type=str, help="Original business zip")
    parser.add_argument("--c_addr", required=True, type=str, help="Corrected business address")
    parser.add_argument("--c_city", required=True, type=str, help="Corrected business city")
    parser.add_argument("--c_zip", required=True, type=str, help="Corrected business zip")
    parser.add_argument("--c_reason", required=False, type=str, help="Reason for the address correction")
    parser.add_argument("--source", required=False, type=str, help="Source of the address correction")

    parser.add_argument("--location_file", help="Path to the locations file (CSV)")
    parser.add_argument("--output", help="Path to the output file for corrected locations (CSV)")

    args = parser.parse_args()

    add_address_correction(
        agtno=args.agtno,
        original_busaddr=args.o_addr,
        original_buscity=args.o_city,
        original_buszip=args.o_zip,
        corrected_busaddr=args.c_addr,
        corrected_buscity=args.c_city,
        corrected_buszip=args.c_zip,
        correction_reason=args.c_reason,
        source=args.source,
    )

    #location_df = pd.read_csv(args.location_file)
    #corrected_locations_df = apply_address_corrections(location_df=location_df)
    #corrected_locations_df.to_csv(args.output, index=False)
