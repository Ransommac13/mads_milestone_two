"""

ingest.py
This module ingests data from the ACS API and writes it to a parquet file.
"""

import pandas as pd

def ingest_acs_data(variable_map: dict, api_key: str | None = None) -> pd.DataFrame:

    return pd.DataFrame()

if __name__ == "__main__":

    ingest_acs_data(variable_map={}, api_key=None)