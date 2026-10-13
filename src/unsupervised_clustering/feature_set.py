"""

feature_set.py

This module defines the feature set by which we want to define a lottery retailer's potential "archetype."
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import TRAIN_DATA_PARQUET_PATH
from src.correlation_test.high_correlation import build_correlation_matrix, flag_high_correlation_matrix

def withhold_variables(train_df: pd.DataFrame) -> pd.DataFrame:

    # Exclude id and location variables.
    clustering_df = train_df.drop(columns=['agtno', 'longitude', 'latitude', 'bg_geoid', 'tract_geoid', 'region'])
    
    # Exclude categorical variables and the target variable for future cluster evaluation.
    clustering_df = clustering_df.drop(columns=['avg_daily_sales', 'chain', 'bustype', 'aadt_functional_class'])

    return clustering_df

def wac_feature_eng(clustering_df: pd.DataFrame) -> pd.DataFrame:

    job_type_dict = {
        "consumer_facing_jobs" : ['wac_cns07', 'wac_cns17', 'wac_cns18', 'wac_cns19'],
        "office_jobs" : ['wac_cns09', 'wac_cns10', 'wac_cns11', 'wac_cns12', 'wac_cns13'],
        "industrial_and_log_jobs" : ['wac_cns01', 'wac_cns02', 'wac_cns03', 'wac_cns04', 'wac_cns05', 'wac_cns06', 'wac_cns08'],
        "supporting_services" : ['wac_cns14', 'wac_cns15', 'wac_cns16', 'wac_cns20']
        }

    for job_type, sectors in job_type_dict.items():

        clustering_df[f"{job_type}_count"] = clustering_df[sectors].sum(axis=1, min_count=len(sectors))

        clustering_df[f"{job_type}_proportion"] = clustering_df[f"{job_type}_count"] / clustering_df['wac_c000'].replace(0, np.nan)

        clustering_df = clustering_df.drop(columns=sectors)
        clustering_df = clustering_df.drop(columns=f"{job_type}_count")

    return clustering_df

def balance_feature_representation(clustering_df: pd.DataFrame) -> pd.DataFrame:
    
    #competition_vars = ['competitors_within_1_mi']
    #employment_vars = ['labor_force_participation_rate', 'unemployment_rate', 'D1C',
    #                   'rac_c000', 'rac_ce01_prop', 'rac_ce02_prop',
    #                   'consumer_facing_jobs_proportion', 'office_jobs_proportion',
    #                   'industrial_and_log_jobs_proportion', 'supporting_services_proportion',
    #                   'wac_c000', 'wac_ce01_prop', 'wac_ce02_prop']
    #activity_vars = ['commercial_places_within_1_mi', 'community_places_within_1_mi']
    #traffic_vars = ['aadt_count', 'aadt_lanes']
    #accessibility_vars = ['dist_to_read_m', 'NatWalkInd', 'D3A']
    #sociodemographic_vars = ['median_household_income', 'total_population', 'age_18_29_proportion', 'age_30_64_proportion',
    #                         'poverty_rate', 'bachelors_degree_proportion_25plus', 'post_grad_degree_proportion_25plus',
    #                         'owner_occupied_proportion', 'family_households_proportion', 'no_vehicle_households_proportion', 'D1B']

    
    cols_to_drop = ['labor_force_participation_rate', 'unemployment_rate', 'rac_ce01_prop', 'rac_ce02_prop',
                    'office_jobs_proportion', 'supporting_services_proportion', 'wac_c000', 'wac_ce02_prop',
                    'aadt_lanes', 'dist_to_read_m', 'aadt_count',
                    'community_places_within_1_mi', 'D1B', 'age_30_64_proportion', 'post_grad_degree_proportion_25plus']

    clustering_df = clustering_df.drop(columns=cols_to_drop)

    return clustering_df

def evaluate_correlation(clustering_df: pd.DataFrame, method="spearman", threshold = 0.75):

    correlation_df = build_correlation_matrix(clustering_df, method=method)
    high_correlation_df = flag_high_correlation_matrix(correlation_df, threshold=threshold)

    return high_correlation_df

def impute_missing_values(clustering_df: pd.DataFrame, imputer=None) -> pd.DataFrame:

    if imputer is None:
        imputer = SimpleImputer(strategy='median')
        imputer.fit(clustering_df)

    imputed_df = pd.DataFrame(imputer.transform(clustering_df), columns=clustering_df.columns, index=clustering_df.index)

    return imputed_df, imputer

def inspect_skewness(imputed_df: pd.DataFrame):

    skewness = imputed_df.skew().sort_values(key=abs, ascending=False)

    return skewness.to_frame()

def transform_skewed_features(imputed_df: pd.DataFrame) -> pd.DataFrame:

    right_skewed_log_transform_vars = ["D1C", "commercial_places_within_1_mi", "competitors_within_1_mi"]
    
    transformed_df = imputed_df.copy()

    for var in right_skewed_log_transform_vars:
        transformed_df[f"{var}_log1p"] = np.log1p(transformed_df[var])
        transformed_df = transformed_df.drop(columns=var)

    return transformed_df

def scale_features(transformed_df: pd.DataFrame) -> pd.DataFrame:

    scaler = StandardScaler()
    
    scaled_df = pd.DataFrame(scaler.fit_transform(transformed_df), columns=transformed_df.columns, index=transformed_df.index)

    return scaled_df


if __name__ == '__main__':

    train_df = pd.read_parquet(TRAIN_DATA_PARQUET_PATH)

    clustering_df = withhold_variables(train_df)
    clustering_df = wac_feature_eng(clustering_df)
    clustering_df = balance_feature_representation(clustering_df)
    imputed_df, imputer = impute_missing_values(clustering_df)

    skewness_df = inspect_skewness(imputed_df)

    transformed_df = transform_skewed_features(imputed_df)
    scaled_df = scale_features(transformed_df)

    clustering_df.to_parquet("src/unsupervised_clustering/eda_data/preimputed_data.parquet")
    imputed_df.to_parquet("src/unsupervised_clustering/eda_data/imputed_data.parquet")
    skewness_df.to_csv("src/unsupervised_clustering/eda_data/skewness.csv")
    scaled_df.to_parquet("src/unsupervised_clustering/eda_data/scaled_data.parquet")

    print(scaled_df.head())
