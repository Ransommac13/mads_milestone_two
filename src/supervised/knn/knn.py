"""

knn.py

Oh no! I'm an idiot, sorry!
"""

from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from scipy.stats import spearmanr

from src.config import TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH, TEST_DATA_PARQUET_PATH
from src.config import TRAIN_PCA_PARQUET_PATH, VAL_PCA_PARQUET_PATH, TEST_PCA_PARQUET_PATH, GMM_PARQUET_DIR

LOG_FEATURES = ["competitors_within_1_mi", "commercial_places_within_1_mi", "community_places_within_1_mi", "aadt_count", 
                "total_population", "D1C", "rac_c000", "D1B", "wac_c000"]

NUMERIC_FEATURES = ["age_18_29_proportion", "age_30_64_proportion", "poverty_rate",
       "bachelors_degree_proportion_25plus", "owner_occupied_proportion", "unemployment_rate",
       "family_households_proportion", "no_vehicle_households_proportion","D3A", "NatWalkInd", "wac_ce01_prop", "wac_ce02_prop",
       "rac_ce01_prop", "rac_ce02_prop", "median_household_income"]

CATEGORICAL_FEATURES = ["bustype", "chain", "region"]

PCA_FEATURES = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
GMM_FEATURES = ['gmm_prob_0', 'gmm_prob_1', 'gmm_prob_2', 'gmm_prob_3']

FEATURE_SETS = {
    "category_only": CATEGORICAL_FEATURES,
    "numeric_only": LOG_FEATURES + NUMERIC_FEATURES,
    "original": LOG_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    "pca": CATEGORICAL_FEATURES + PCA_FEATURES,
    "gmm": CATEGORICAL_FEATURES + GMM_FEATURES,
    "original_pca_gmm": LOG_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES + PCA_FEATURES + GMM_FEATURES
}

TARGET = 'avg_daily_sales'
OUTPUT_DIR = Path("src/supervised/outputs/model outputs")

def split_data(df):
    """Separate raw features and log-transformed target."""
    X = df.drop(columns=[TARGET])
    y = np.log1p(df[TARGET])
    return X, y

def build_knn_pipeline(feature_columns, categorical_columns, log_columns):
    """
    Create a KNN regression pipeline.

    All preprocessing is fitted within the pipeline,
    preventing leakage across CV folds.
    """

    cat_cols = [c for c in feature_columns if c in categorical_columns]
    log_cols = [c for c in feature_columns if c in log_columns]
    other_num_cols = [c for c in feature_columns if c not in cat_cols + log_cols]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(min_frequency=30, handle_unknown="infrequent_if_exist", sparse_output=False))
        ])

    log_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log_transform", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler())
        ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformers = []

    if cat_cols:
        transformers.append(("categorical", categorical_pipeline, cat_cols))
    if log_cols:
        transformers.append(("log_numeric", log_pipeline, log_cols))
    if other_num_cols:
        transformers.append(("numeric", numeric_pipeline, other_num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", KNeighborsRegressor(algorithm="auto", n_jobs=1))])

    return pipeline


def tune_knn(X_train, y_train, train_groups,
             feature_columns, categorical_columns, log_columns,
             param_grid=None, n_splits=5):
    """
    Tune KNN using GroupKFold on census tract IDs.

    y_train must contain log1p-transformed sales.
    """

    pipeline = build_knn_pipeline(feature_columns, categorical_columns, log_columns)

    if param_grid is None:
        param_grid = {
            "model__n_neighbors": [5, 15, 30, 60, 100],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        }

    cv = GroupKFold(n_splits=n_splits)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    search.fit(
        X_train[feature_columns],
        np.asarray(y_train).ravel(),
        groups=np.asarray(train_groups).ravel()
    )

    return search


def evaluate_knn(model, X, y_log):
    """
    Evaluate fitted model on log and dollar scales.
    """

    y_log = np.asarray(y_log).ravel()
    pred_log = np.asarray(model.predict(X)).ravel()

    y_sales = np.expm1(y_log)
    pred_sales = np.maximum(np.expm1(pred_log), 0)

    return {
        "r2_log": r2_score(y_log, pred_log),
        "mae_log": mean_absolute_error(y_log, pred_log),
        "rmse_log": np.sqrt(mean_squared_error(y_log, pred_log)),
        "spearman_log": spearmanr(y_log, pred_log).statistic,
        "r2_sales": r2_score(y_sales, pred_sales),
        "mae_sales": mean_absolute_error(y_sales, pred_sales),
        "rmse_sales": np.sqrt(mean_squared_error(y_sales, pred_sales)),
        "spearman_sales": spearmanr(y_sales, pred_sales).statistic
    }


def run_knn_experiments(X_train, y_train, X_val, y_val, feature_sets,
                        categorical_columns, log_columns, group_column="tract_geoid", param_grid=None):
    """
    Tune and evaluate KNN for each feature set.

    Returns
    results_df : of cross-validation and validation metrics.
    fitted_models : dict of best fitted pipelines for each feature set.
    """

    results = []
    fitted_models = {}

    for name, features in feature_sets.items():

        features = list(dict.fromkeys(features))

        print(f"\nTraining KNN: {name}")

        search = tune_knn(
            X_train=X_train,
            y_train=y_train,
            train_groups=X_train[group_column],
            feature_columns=features,
            categorical_columns=categorical_columns,
            log_columns=log_columns,
            param_grid=param_grid
        )

        best_model = search.best_estimator_
        fitted_models[name] = best_model

        train_metrics = evaluate_knn(best_model, X_train[features], y_train)
        val_metrics = evaluate_knn(best_model, X_val[features], y_val)

        results.append({
            "feature_set": name,
            "model": "knn",
            "best_params": search.best_params_,
            "cv_r2_log": search.best_score_,

            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },

            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
            }
        })

    results_df = pd.DataFrame(results)

    return results_df, fitted_models

if __name__ == '__main__':

    X_train_orig = pd.read_parquet(TRAIN_DATA_PARQUET_PATH)
    X_train_pca = pd.read_parquet(TRAIN_PCA_PARQUET_PATH)
    gmm_dir = Path(GMM_PARQUET_DIR)
    X_train_gmm = pd.read_parquet(gmm_dir / "X_gmm_train.parquet")[['agtno', 'gmm_prob_0', 'gmm_prob_1', 'gmm_prob_2', 'gmm_prob_3']]

    X_val_orig = pd.read_parquet(VALIDATION_DATA_PARQUET_PATH)
    X_val_pca = pd.read_parquet(VAL_PCA_PARQUET_PATH)
    gmm_dir = Path(GMM_PARQUET_DIR)
    X_val_gmm = pd.read_parquet(gmm_dir / "X_gmm_val.parquet")[['agtno', 'gmm_prob_0', 'gmm_prob_1', 'gmm_prob_2', 'gmm_prob_3']]

    X_train_full = X_train_orig.merge(X_train_pca, on='agtno', how='inner', validate='one_to_one')
    X_train_full = X_train_full.merge(X_train_gmm, on='agtno', how='inner', validate='one_to_one')
    
    X_val_full = X_val_orig.merge(X_val_pca, on='agtno', how='inner', validate='one_to_one')
    X_val_full = X_val_full.merge(X_val_gmm, on='agtno', how='inner', validate='one_to_one')

    X_train, y_train = split_data(X_train_full)
    X_val, y_val = split_data(X_val_full)
    
    results_df, fitted_models = run_knn_experiments(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_sets=FEATURE_SETS,
        categorical_columns=CATEGORICAL_FEATURES,
        log_columns=LOG_FEATURES
    )

    print(results_df.round(4).to_string(index=False))

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "knn_results.csv", index=False)

    # Select the feature set with the highest CV R²
    best_feature_set = (results_df.loc[results_df["cv_r2_log"].idxmax(), "feature_set"])
    best_model = fitted_models[best_feature_set]
    joblib.dump(best_model, output_dir / "knn_best_model.joblib")

    print("Best feature set:", best_feature_set)


