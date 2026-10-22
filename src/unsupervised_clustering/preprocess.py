"""

preprocess.py

This module creates a pipeline to preprocess the data for modeling.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib

from src.config import TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH, TEST_DATA_PARQUET_PATH
from src.config import TRAIN_PCA_PARQUET_PATH, VAL_PCA_PARQUET_PATH, TEST_PCA_PARQUET_PATH
from src.unsupervised_clustering.helper_functions import scree_plot, pc_loading_heatmap

CLUSTERING_FEATURES = ['competitors_within_1_mi', 'median_household_income',
       'total_population', 'age_18_29_proportion', 'poverty_rate',
       'bachelors_degree_proportion_25plus', 'owner_occupied_proportion',
       'family_households_proportion', 'no_vehicle_households_proportion',
       'D1C', 'D3A', 'NatWalkInd', 'commercial_places_within_1_mi', 'rac_c000',
       'wac_ce01_prop', 'consumer_facing_jobs_proportion',
       'industrial_and_log_jobs_proportion', 'aadt_count']

def wac_feature_eng(clustering_df: pd.DataFrame) -> pd.DataFrame:

    clustering_df = clustering_df.copy()
    
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

def select_features(clustering_df: pd.DataFrame):

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

    return clustering_df[CLUSTERING_FEATURES].copy()

def build_preprocessing_pipeline(feature_cols):

    log_transform_cols = ["D1C", "commercial_places_within_1_mi", "competitors_within_1_mi", "aadt_count"]
    other_cols = [col for col in feature_cols if col not in log_transform_cols]

    log_transform_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log_transform', FunctionTransformer(np.log1p, feature_names_out='one-to-one'))
    ])

    impute_only_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    column_transformer = ColumnTransformer([
        ('log_transform_cols', log_transform_pipeline, log_transform_cols),
        ('other_cols', impute_only_pipeline, other_cols)
    ])

    preprocessing_pipeline = Pipeline([
        ('transform', column_transformer),
        ('scaler', StandardScaler())
    ])

    return preprocessing_pipeline

def main_pipepline(X_train_df: pd.DataFrame):
    
    full_preprocessor = Pipeline([
        ('wac_engineering', FunctionTransformer(wac_feature_eng, validate=False)),
        ('feature_selection', FunctionTransformer(select_features)),
        ('numeric_preprocessing', build_preprocessing_pipeline(CLUSTERING_FEATURES))
    ])

    X_train_processed = full_preprocessor.fit_transform(X_train_df)
    
    return full_preprocessor, X_train_processed

def fit_PCA(X_train_processed: pd.DataFrame, agtno):

    pca = PCA()

    assert X_train_processed.index.equals(train_df.index)
    assert agtno.is_unique

    X_pca = pca.fit_transform(X_train_processed)

    X_pca = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=X_train_processed.index
    )

    X_pca["agtno"] = agtno.reindex(X_pca.index)
    assert X_pca['agtno'].notna().all()

    feature_names = X_train_processed.columns

    return pca, X_pca, feature_names

def transform_PCA(pca, processed_df: pd.DataFrame, agtno):

    X_pca = pd.DataFrame(
        pca.transform(processed_df),
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=processed_df.index
    )

    X_pca["agtno"] = agtno.reindex(X_pca.index)
    assert X_pca['agtno'].notna().all()

    return X_pca

if __name__ == "__main__":

    train_df = pd.read_parquet(TRAIN_DATA_PARQUET_PATH)
    val_df = pd.read_parquet(VALIDATION_DATA_PARQUET_PATH)
    test_df = pd.read_parquet(TEST_DATA_PARQUET_PATH)
    
    preprocessor, X_train_processed = main_pipepline(train_df)
    X_val_processed = preprocessor.transform(val_df)
    X_test_processed = preprocessor.transform(test_df)
    #joblib.dump(preprocessor, "src/unsupervised_clustering/fitted_objects/preprocess.joblib")

    column_names = preprocessor.named_steps['numeric_preprocessing'].get_feature_names_out()
    new_column_names = np.array(["log1p" + col[19:] if col.startswith("log_transform_cols") else col for col in column_names])
    new_column_names = np.array([col[12:] if col.startswith("other_cols") else col for col in new_column_names])
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=new_column_names, index=train_df.index)
    X_val_processed_df = pd.DataFrame(X_val_processed, columns=new_column_names, index=val_df.index)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=new_column_names, index=test_df.index)

    #X_train_processed_df.to_parquet("src/unsupervised_clustering/eda_data/X_train_processed.parquet")
    #X_val_processed_df.to_parquet("src/unsupervised_clustering/eda_data/X_val_processed.parquet")
    #X_test_processed_df.to_parquet("src/unsupervised_clustering/eda_data/X_test_processed.parquet")

    #print("Preprocessed datasets written to parquet!")
    print(f"Train shape: {X_train_processed_df.shape}. Val shape: {X_val_processed.shape}. Test shape: {X_test_processed.shape}")

    pca, X_train_pca, feature_names = fit_PCA(X_train_processed_df, train_df["agtno"])
    X_val_pca = transform_PCA(pca, X_val_processed_df, val_df["agtno"])
    X_test_pca = transform_PCA(pca, X_test_processed_df, test_df["agtno"])
    #joblib.dump(pca, "src/unsupervised_clustering/fitted_objects/pca.joblib")

    X_pca_train_path = Path(TRAIN_PCA_PARQUET_PATH)
    X_pca_val_path = Path(VAL_PCA_PARQUET_PATH)
    X_pca_test_path = Path(TEST_PCA_PARQUET_PATH)
    X_pca_train_path.parent.mkdir(parents=True, exist_ok=True)
    #X_train_pca.to_parquet(str(X_pca_train_path), index=False, engine="pyarrow")
    #X_val_pca.to_parquet(str(X_pca_val_path), index=False, engine="pyarrow")
    #X_test_pca.to_parquet(str(X_pca_test_path), index=False, engine="pyarrow")

    #print("PCs written to parquet!")
    print(f"Train PCs shape: {X_train_pca.shape}. Val PCs shape: {X_val_pca.shape}. Test PCs shape: {X_test_pca.shape}")

    #scree_plot(pca)
    
    #pc_loadings = pd.DataFrame(pca.components_.T, index=feature_names, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    #pc_loading_heatmap(pc_loadings)
