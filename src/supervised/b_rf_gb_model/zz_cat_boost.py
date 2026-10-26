"""

cat_boost.py

Hail Mary full of Grace! I do not believe but help me anyways!
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

from src.config import RANDOM_SEED, TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH
from src.supervised.b_rf_gb_model.b1_model_supervised import OUTPUT

TARGET = "avg_daily_sales"

LOG_FEATURES = ["competitors_within_1_mi", "commercial_places_within_1_mi", "community_places_within_1_mi", "aadt_count", 
                "total_population", "D1C", "rac_c000", "D1B", "wac_c000"]

NUMERIC_FEATURES = ["age_18_29_proportion", "age_30_64_proportion", "poverty_rate",
       "bachelors_degree_proportion_25plus", "owner_occupied_proportion", "unemployment_rate",
       "family_households_proportion", "no_vehicle_households_proportion","D3A", "NatWalkInd", "wac_ce01_prop", "wac_ce02_prop",
       "rac_ce01_prop", "rac_ce02_prop", "median_household_income"]

CATEGORICAL_FEATURES = ["bustype", "chain", "region"]

ALL_NUMERIC_FEATURES = LOG_FEATURES + NUMERIC_FEATURES
ALL_FEATURES = LOG_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():

    train_df = pd.read_parquet(TRAIN_DATA_PARQUET_PATH)
    val_df = pd.read_parquet(VALIDATION_DATA_PARQUET_PATH)

    return train_df, val_df

def split_data(df):
    """Separate raw features and log-transformed target."""
    X = df.drop(columns=[TARGET])
    y = np.log1p(df[TARGET])
    return X, y

def prepare_catboost_data(X, feature_columns, categorical_columns):
    """
    Select features and prepare categorical variables for CatBoost's native categorical handling.

    Input:
        X: Original, unprocessed DataFrame
        feature_columns: All selected model features
        categorical_columns: Selected categorical features

    Returns:
        DataFrame ready for CatBoost
    """
    X_out = X.loc[:, feature_columns].copy()

    for col in categorical_columns:
        X_out[col] = X_out[col].fillna("__MISSING__").astype(str)

    numeric_columns = [col for col in feature_columns if col not in categorical_columns]

    for col in numeric_columns:
        X_out[col] = pd.to_numeric(X_out[col], errors="raise")
        X_out[col] = X_out[col].replace([np.inf, -np.inf], np.nan)

    return X_out

def build_catboost_model(random_state=RANDOM_SEED):
    """
    Initialize a CatBoost regressor.
    Target is expected to be log1p(sales).
    """
    catboost_regressor = CatBoostRegressor(
        loss_function="RMSE",
        iterations=800,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=10,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        thread_count=1
    )

    return catboost_regressor

def tune_catboost(X_train, y_train, train_groups, categorical_columns,
                  param_grid=None, n_splits=5, random_state=RANDOM_SEED):
    """
    Tune CatBoost using tract-aware cross-validation.

    X_train should already be prepared.
    y_train should be log1p(sales).
    train_groups should contain tract_geoid.
    """

    if param_grid is None:
        param_grid = {
            "depth": [4, 6, 8],
            "l2_leaf_reg": [3, 10],
            "iterations": [500, 1000]
        }

    model = build_catboost_model(random_state=random_state)

    cv = GroupKFold(n_splits=n_splits)

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    search.fit(X_train, np.asarray(y_train).ravel(), groups=np.asarray(train_groups).ravel(), cat_features=categorical_columns)

    return search

def evaluate_catboost(model, X, y_log):
    """
    Evaluate model on both log and original sales scales.

    Returns a dictionary of performance metrics.
    """

    y_log = np.asarray(y_log).ravel()
    pred_log = model.predict(X)

    y_sales = np.expm1(y_log)
    pred_sales = np.maximum(np.expm1(pred_log), 0)

    results = {
        "r2_log": r2_score(y_log, pred_log),
        "rmse_log": np.sqrt(mean_squared_error(y_log, pred_log)),
        "mae_log": mean_absolute_error(y_log, pred_log),
        "spearman": spearmanr(y_log, pred_log).statistic,
        "r2_sales": r2_score(y_sales, pred_sales),
        "rmse_sales": np.sqrt(mean_squared_error(y_sales, pred_sales)),
        "mae_sales": mean_absolute_error(y_sales, pred_sales)
    }

    return results

def main(X_train, y_train, X_val, y_val):

    feature_sets = {
        'cat_only': CATEGORICAL_FEATURES,
        'orig_no_region': ALL_NUMERIC_FEATURES + ['bustype', 'chain'],
        'all_features': ALL_FEATURES
    }

    results = []
    fitted_models = {}

    for name, features in feature_sets.items():

        categorical = [col for col in CATEGORICAL_FEATURES if col in features]

        X_tr = prepare_catboost_data(X_train, features, categorical)
        X_v = prepare_catboost_data(X_val, features, categorical)

        # Hyperparameter selection on training data only
        search = tune_catboost(
            X_train=X_tr,
            y_train=y_train,
            train_groups=X_train['tract_geoid'],
            categorical_columns=categorical
        )

        best_model = search.best_estimator_
        fitted_models[name] = best_model

        # Evaluate best model
        train_metrics = evaluate_catboost(best_model, X_tr, y_train)
        val_metrics = evaluate_catboost(best_model, X_v, y_val)

        results.append({
            "feature_set": name,
            "model": "catboost",
            "best_params": search.best_params_,
            "cv_r2_log": search.best_score_,
            "train_r2_log": train_metrics["r2_log"],
            "val_r2_log": val_metrics["r2_log"],
            "val_mae_log": val_metrics["mae_log"],
            "val_spearman": val_metrics["spearman"],
            "val_mae_sales": val_metrics["mae_sales"]
        })

    results_df = pd.DataFrame(results)

    return results_df, fitted_models

def evaluate_by_group(y_true_log, y_pred_log, groups, min_group_size=10):
    """
    Evaluate regression performance within categorical groups.

    Parameters
    y_true_log : Actual log1p(sales).
    y_pred_log : Predicted log1p(sales).
    groups : Group membership for each observation, such as bustype, region, or chain.
    min_group_size : Minimum observations required to report a group's metrics. Smaller groups are excluded.

    Returns group-level performance metrics, sorted by MAE on the log scale (worst first).
    """

    y_true_log = np.asarray(y_true_log).ravel()
    y_pred_log = np.asarray(y_pred_log).ravel()
    groups = np.asarray(groups)

    if not (len(y_true_log) == len(y_pred_log) == len(groups)):
        raise ValueError("Inputs must have the same length.")

    df = pd.DataFrame({
        "group": pd.Series(groups).fillna("__MISSING__"),
        "actual_log": y_true_log,
        "pred_log": y_pred_log
    })

    df["actual_sales"] = np.expm1(df["actual_log"])
    df["pred_sales"] = np.maximum(np.expm1(df["pred_log"]), 0)

    df["error_log"] = df["pred_log"] - df["actual_log"]
    df["error_sales"] = (df["pred_sales"] - df["actual_sales"])

    results = []

    for name, subset in df.groupby("group"):

        n = len(subset)
        if n < min_group_size:
            continue

        results.append({
            "group": name,
            "n_retailers": n,
            "r2_log": r2_score(subset["actual_log"], subset["pred_log"] if n >= 2 else np.nan),
            "mae_log": mean_absolute_error(subset["actual_log"], subset["pred_log"]),
            "mae_sales": mean_absolute_error(subset["actual_sales"], subset["pred_sales"]),
            "normalized_mae_sales": mean_absolute_error(subset["actual_sales"], subset["pred_sales"]) / subset["actual_sales"].mean() if subset["actual_sales"].mean() > 0 else np.nan, 
            "bias_log": subset["error_log"].mean(),
            "bias_sales": subset["error_sales"].mean(),
            "mean_actual_sales": (subset["actual_sales"].mean()),
            "mean_pred_sales": (subset["pred_sales"].mean())
        })

    results_df = pd.DataFrame(results).sort_values("mae_log", ascending=False).reset_index(drop=True)

    return results_df

if __name__ == '__main__':

    
    train_df, val_df = load_data()
    X_train, y_train = split_data(train_df)
    X_val, y_val = split_data(val_df)
    
    #results, fitted_models = main(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    #best_model = fitted_models['all_features']

    best_model = CatBoostRegressor()
    best_model.load_model(str(OUTPUT / "catboost_all_features.cbm"))
    X_v = prepare_catboost_data(X_val, ALL_FEATURES, CATEGORICAL_FEATURES)
    y_pred_log = best_model.predict(X_v)

    bustype_results = evaluate_by_group(y_true_log=y_val, y_pred_log=y_pred_log, groups=X_val["bustype"], min_group_size=20)
    region_results = evaluate_by_group(y_true_log=y_val, y_pred_log=y_pred_log, groups=X_val["region"], min_group_size=20)
    chain_results = evaluate_by_group(y_true_log=y_val, y_pred_log=y_pred_log, groups=X_val["chain"], min_group_size=20)

    #OUTPUT.mkdir(parents=True, exist_ok=True)
    #best_model.save_model(str(OUTPUT / "catboost_all_features.cbm"))
    #results.to_csv(str(OUTPUT / "catboost_results.csv"))
    #metadata = {
    #    'features': ALL_FEATURES,
    #    'cat_features': CATEGORICAL_FEATURES, 
    #    'cv_group': 'tract_geoid',
    #    'best_params': best_model.get_params()
    #}
    #with open(str(OUTPUT / "catboost_metadata.json"), "w") as f:
    #    json.dump(metadata, f, indent=2)
