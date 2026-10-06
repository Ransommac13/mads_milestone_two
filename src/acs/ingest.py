"""

ingest.py
This module ingests data from the ACS API and writes it to a parquet file.
"""

import requests
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import ACS_API_KEY, ACS_BASE_URL, ACS_INTERIM_PARQUET_PATH, ACS_PARQUET_PATH, FINAL_LOCATION_PARQUET_PATH

NYS_FIPS = "36"

ACS_AGE_VARIABLES = {
    # Male 18-24
    "B01001_007E": "male_18_19",
    "B01001_008E": "male_20",
    "B01001_009E": "male_21",
    "B01001_010E": "male_22_24",

    # Male 25-44
    "B01001_011E": "male_25_29",
    "B01001_012E": "male_30_34",
    "B01001_013E": "male_35_39",
    "B01001_014E": "male_40_44",

    # Male 45-64
    "B01001_015E": "male_45_49",
    "B01001_016E": "male_50_54",
    "B01001_017E": "male_55_59",
    "B01001_018E": "male_60_61",
    "B01001_019E": "male_62_64",

    # Male 65+
    "B01001_020E": "male_65_66",
    "B01001_021E": "male_67_69",
    "B01001_022E": "male_70_74",
    "B01001_023E": "male_75_79",
    "B01001_024E": "male_80_84",
    "B01001_025E": "male_85_plus",

    # Female 18-24
    "B01001_031E": "female_18_19",
    "B01001_032E": "female_20",
    "B01001_033E": "female_21",
    "B01001_034E": "female_22_24",

    # Female 25-44
    "B01001_035E": "female_25_29",
    "B01001_036E": "female_30_34",
    "B01001_037E": "female_35_39",
    "B01001_038E": "female_40_44",

    # Female 45-64
    "B01001_039E": "female_45_49",
    "B01001_040E": "female_50_54",
    "B01001_041E": "female_55_59",
    "B01001_042E": "female_60_61",
    "B01001_043E": "female_62_64",

    # Female 65+
    "B01001_044E": "female_65_66",
    "B01001_045E": "female_67_69",
    "B01001_046E": "female_70_74",
    "B01001_047E": "female_75_79",
    "B01001_048E": "female_80_84",
    "B01001_049E": "female_85_plus"
}

ACS_AGE_MOES = {
    # Male 18-24
    "B01001_007M": "male_18_19_moe",
    "B01001_008M": "male_20_moe",
    "B01001_009M": "male_21_moe",
    "B01001_010M": "male_22_24_moe",

    # Male 25-44
    "B01001_011M": "male_25_29_moe",
    "B01001_012M": "male_30_34_moe",
    "B01001_013M": "male_35_39_moe",
    "B01001_014M": "male_40_44_moe",

    # Male 45-64
    "B01001_015M": "male_45_49_moe",
    "B01001_016M": "male_50_54_moe",
    "B01001_017M": "male_55_59_moe",
    "B01001_018M": "male_60_61_moe",
    "B01001_019M": "male_62_64_moe",

    # Male 65+
    "B01001_020M": "male_65_66_moe",
    "B01001_021M": "male_67_69_moe",
    "B01001_022M": "male_70_74_moe",
    "B01001_023M": "male_75_79_moe",
    "B01001_024M": "male_80_84_moe",
    "B01001_025M": "male_85_plus_moe",

    # Female 18-24
    "B01001_031M": "female_18_19_moe",
    "B01001_032M": "female_20_moe",
    "B01001_033M": "female_21_moe",
    "B01001_034M": "female_22_24_moe",

    # Female 25-44
    "B01001_035M": "female_25_29_moe",
    "B01001_036M": "female_30_34_moe",
    "B01001_037M": "female_35_39_moe",
    "B01001_038M": "female_40_44_moe",

    # Female 45-64
    "B01001_039M": "female_45_49_moe",
    "B01001_040M": "female_50_54_moe",
    "B01001_041M": "female_55_59_moe",
    "B01001_042M": "female_60_61_moe",
    "B01001_043M": "female_62_64_moe",

    # Female 65+
    "B01001_044M": "female_65_66_moe",
    "B01001_045M": "female_67_69_moe",
    "B01001_046M": "female_70_74_moe",
    "B01001_047M": "female_75_79_moe",
    "B01001_048M": "female_80_84_moe",
    "B01001_049M": "female_85_plus_moe"
}

ACS_SOCIOECONOMIC_VARIABLES = {
    "B19013_001E": "median_household_income",
    "B19013_001M": "median_household_income_moe",
    "B17101_001E": "poverty_universe",
    "B17101_001M": "poverty_universe_moe",
    "B17101_002E": "below_poverty",
    "B17101_002M": "below_poverty_moe"
}

ACS_EDUCATION_VARIABLES = {
    "B15003_001E": "education_25plus_total",
    "B15003_001M": "education_25plus_total_moe",
    "B15003_022E": "bachelors_degree",
    "B15003_022M": "bachelors_degree_moe",
    "B15003_023E": "masters_degree",
    "B15003_023M": "masters_degree_moe",
    "B15003_024E": "professional_degree",
    "B15003_024M": "professional_degree_moe",
    "B15003_025E": "doctorate_degree",
    "B15003_025M": "doctorate_degree_moe"
}

ACS_EMPLOYMENT_VARIABLES = {
    "B23025_001E": "population_16plus",
    "B23025_001M": "population_16plus_moe",
    "B23025_002E": "in_labor_force",
    "B23025_002M": "in_labor_force_moe",
    "B23025_005E": "unemployed",
    "B23025_005M": "unemployed_moe"
}

ACS_HOUSING_VARIABLES = {
    "B25003_001E": "occupied_housing_units",
    "B25003_001M": "occupied_housing_units_moe",
    "B25003_002E": "owner_occupied",
    "B25003_002M": "owner_occupied_moe",
    "B25003_003E": "renter_occupied",
    "B25003_003M": "renter_occupied_moe"
}

ACS_HOUSEHOLD_VARIABLES = {
    "B11001_001E": "households_total",
    "B11001_001M": "households_total_moe",
    "B11001_002E": "family_households",
    "B11001_002M": "family_households_moe",
    "B11001_007E": "nonfamily_households",
    "B11001_007M": "nonfamily_households_moe",
    "B11001_008E": "living_alone_households",
    "B11001_008M": "living_alone_households_moe"
}

def ingest_acs_data(variable_map: dict, api_key: str | None = None, include_name: bool = False) -> pd.DataFrame:

    if api_key is None:
        api_key = ACS_API_KEY

    if not api_key: 
        raise ValueError("ACS API key is missing. Please provide it as an argument or set it in the .env file.")

    census_variables = list(variable_map.keys())
    get_variables = census_variables.copy()

    if include_name:
        get_variables.insert(0, "NAME")

    params = {
        "get": ",".join(get_variables),
        "for": "block group:*",
        "in": [f"state:{NYS_FIPS}", "county:*", "tract:*"],
        "key": api_key
    }

    response = requests.get(ACS_BASE_URL, params=params, timeout=60)
    response.raise_for_status()

    #print(response.url)
    #print(response.status_code)
    #print(response.text[:2000])

    data = response.json()
    feature_df = pd.DataFrame(data[1:], columns=data[0])

    feature_df['bg_geoid'] = feature_df['state'] + feature_df['county'] + feature_df['tract'] + feature_df['block group']

    for variable in census_variables:
        feature_df[variable] = pd.to_numeric(feature_df[variable], errors='coerce')

    feature_df = feature_df.rename(columns=variable_map)

    output_cols = ["bg_geoid"]
    if include_name:
        output_cols.append("NAME")
    output_cols.extend(variable_map.values())

    return feature_df[output_cols]

def aggregate_counts(all_features_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates the ACS age counts into broader age groups."""
    # Define the aggregation mapping
    aggregation_mapping = {
        "age_18_29": ["male_18_19", "male_20", "male_21", "male_22_24", "male_25_29", "female_18_19", "female_20", "female_21", "female_22_24", "female_25_29"],
        "age_30_64": ["male_30_34", "male_35_39", "male_40_44", "male_45_49", "male_50_54", "male_55_59", "male_60_61", "male_62_64", "female_30_34", "female_35_39", "female_40_44", "female_45_49", "female_50_54", "female_55_59", "female_60_61", "female_62_64"],
        "age_65_plus": ["male_65_66", "male_67_69", "male_70_74", "male_75_79", "male_80_84", "male_85_plus", "female_65_66", "female_67_69", "female_70_74", "female_75_79", "female_80_84", "female_85_plus"],
        "post_grad_degree": ["masters_degree", "professional_degree", "doctorate_degree"]
    }

    # Perform the aggregation
    for new_col, old_cols in aggregation_mapping.items():
        all_features_df[new_col] = all_features_df[old_cols].sum(axis=1)
        all_features_df = all_features_df.drop(columns=old_cols)

    return all_features_df

def aggregate_moes(all_features_df: pd.DataFrame) -> pd.DataFrame:

    """Aggregates the ACS age margins of error (MOEs) into broader age groups."""
    # Define the aggregation mapping
    aggregation_mapping = {
        "age_18_29_moe": ["male_18_19_moe", "male_20_moe", "male_21_moe", "male_22_24_moe", "male_25_29_moe", "female_18_19_moe", "female_20_moe", "female_21_moe", "female_22_24_moe", "female_25_29_moe"],
        "age_30_64_moe": ["male_30_34_moe", "male_35_39_moe", "male_40_44_moe", "male_45_49_moe", "male_50_54_moe", "male_55_59_moe", "male_60_61_moe", "male_62_64_moe", "female_30_34_moe", "female_35_39_moe", "female_40_44_moe", "female_45_49_moe", "female_50_54_moe", "female_55_59_moe", "female_60_61_moe", "female_62_64_moe"],
        "age_65_plus_moe": ["male_65_66_moe", "male_67_69_moe", "male_70_74_moe", "male_75_79_moe", "male_80_84_moe", "male_85_plus_moe", "female_65_66_moe", "female_67_69_moe", "female_70_74_moe", "female_75_79_moe", "female_80_84_moe", "female_85_plus_moe"],
        "post_grad_degree_moe": ["masters_degree_moe", "professional_degree_moe", "doctorate_degree_moe"]
    }

    # Perform the aggregation using the square root of the sum of squares method
    for new_col, old_cols in aggregation_mapping.items():
        all_features_df[new_col] = np.sqrt((all_features_df[old_cols] ** 2).sum(axis=1))
        all_features_df = all_features_df.drop(columns=old_cols)

    return all_features_df

def clean_acs_data(all_acs_features: pd.DataFrame) -> pd.DataFrame:
    """Cleans the ACS data."""
    
    # Socioeconomic data cleaning
    SOCIOECONOMIC_SENTINELS = [-666666666,-222222222, -333333333]
    all_acs_features['median_household_income'] = all_acs_features['median_household_income'].replace(SOCIOECONOMIC_SENTINELS, np.nan)
    all_acs_features['median_household_income_moe'] = all_acs_features['median_household_income_moe'].replace(SOCIOECONOMIC_SENTINELS, np.nan)

    
    return all_acs_features

def feature_eng_acs_data(all_acs_features: pd.DataFrame) -> pd.DataFrame:
    """Performs feature engineering on the ACS data."""

    # Calculate total population for each block group
    all_acs_features['total_population'] = all_acs_features[['age_18_29', 'age_30_64', 'age_65_plus']].sum(axis=1)

    # Calculate the proportion of each age group relative to the total population.
    for age_group in ['age_18_29', 'age_30_64', 'age_65_plus']:
        proportion_col = f"{age_group}_proportion"
        all_acs_features[proportion_col] = all_acs_features[age_group] / all_acs_features['total_population']

    # Calculate the poverty rate.
    all_acs_features['poverty_rate'] = all_acs_features['below_poverty'] / all_acs_features['poverty_universe']

    # Calculate the proportion of each education gruop relative to the total population.
    for ed_group in ['bachelors_degree', 'post_grad_degree']:
        proportion_col = f"{ed_group}_proportion_25plus"
        all_acs_features[proportion_col] = all_acs_features[ed_group] / all_acs_features['education_25plus_total']

    # Calculate labor force participation and unemployment rates.
    for emp_stat in ['in_labor_force', 'unemployed']:
        proportion_col = f"{emp_stat}_proportion_16plus"
        all_acs_features[proportion_col] = all_acs_features[emp_stat] / all_acs_features['population_16plus']

    # Calculate proporations of owner- and renter-occupied housing units.
    for housing_status in ['owner_occupied', 'renter_occupied']:
        proportion_col = f"{housing_status}_proportion"
        all_acs_features[proportion_col] = all_acs_features[housing_status] / all_acs_features['occupied_housing_units']

    # Calculate proportions of household types.
    all_acs_features['family_households_proportion'] = all_acs_features['family_households'] / all_acs_features['households_total']
    all_acs_features['nonfamily_multiperson_households_proportion'] = (all_acs_features['nonfamily_households'] / all_acs_features['households_total']) - (all_acs_features['living_alone_households'] / all_acs_features['households_total'])
    all_acs_features['living_alone_households_proportion'] = all_acs_features['living_alone_households'] / all_acs_features['households_total']
    
    return all_acs_features

def trim_to_desired_features(all_acs_features: pd.DataFrame) -> pd.DataFrame:

    moe_cols = [col for col in all_acs_features.columns if col.endswith("_moe")]
    cols_used_for_eng = ['age_18_29', 'age_30_64', 'age_65_plus', 'poverty_universe', 'below_poverty',
                         'education_25plus_total', 'bachelors_degree', 'post_grad_degree',
                         'in_labor_force', 'unemployed', 'population_16plus',
                         'owner_occupied', 'renter_occupied', 'occupied_housing_units',
                         'family_households', 'nonfamily_households', 'living_alone_households']
    
    cols_to_drop = moe_cols + cols_used_for_eng

    acs_training_features = all_acs_features.drop(columns=cols_to_drop)

    return acs_training_features


if __name__ == "__main__":

    acs_age_counts = ingest_acs_data(variable_map=ACS_AGE_VARIABLES, api_key=None)
    acs_age_moes = ingest_acs_data(variable_map=ACS_AGE_MOES, api_key=None)
    
    acs_socioeconomic = ingest_acs_data(variable_map=ACS_SOCIOECONOMIC_VARIABLES, api_key=None)
    acs_education = ingest_acs_data(variable_map=ACS_EDUCATION_VARIABLES, api_key=None)
    acs_employment = ingest_acs_data(variable_map=ACS_EMPLOYMENT_VARIABLES, api_key=None)
    acs_housing = ingest_acs_data(variable_map=ACS_HOUSING_VARIABLES, api_key=None)
    acs_household = ingest_acs_data(variable_map=ACS_HOUSEHOLD_VARIABLES, api_key=None)

    age_features_df = pd.merge(acs_age_counts, acs_age_moes, on='bg_geoid', how='left')
    other_acs_features_df = acs_socioeconomic.merge(acs_education, on='bg_geoid', how='left') \
        .merge(acs_employment, on='bg_geoid', how='left') \
        .merge(acs_housing, on='bg_geoid', how='left') \
        .merge(acs_household, on='bg_geoid', how='left')
    
    raw_features_df = age_features_df.merge(other_acs_features_df, on='bg_geoid', how='left')
    aggregated_features_df = aggregate_counts(raw_features_df)
    aggregated_features_df = aggregate_moes(aggregated_features_df)
    cleaned_features_df = clean_acs_data(aggregated_features_df)
    acs_all_features_df = feature_eng_acs_data(cleaned_features_df)
    acs_training_features_df = trim_to_desired_features(acs_all_features_df)

    locations_df = pd.read_parquet(FINAL_LOCATION_PARQUET_PATH)
    location_data_acs_df = locations_df.merge(acs_training_features_df, on="bg_geoid", how="left", validate="many_to_one")

    acs_interim_parquet_path = Path(ACS_INTERIM_PARQUET_PATH)
    acs_interim_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    acs_all_features_df.to_parquet(str(acs_interim_parquet_path), index=False, engine="pyarrow")
    
    
    acs_parquet_path = Path(ACS_PARQUET_PATH)
    acs_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    location_data_acs_df.to_parquet(str(acs_parquet_path), index=False, engine="pyarrow")

    print("ACS features written to parquet!")
