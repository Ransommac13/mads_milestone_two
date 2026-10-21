"""

spatial_diagnostics.py

Analyze the spatial autocorrelation of residual errors
to determine whether a retailer's physical location can represent uncaptured predictive information.
"""

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
import geopandas as gpd
from libpysal.weights import KNN
from esda.moran import Moran

from src.config import RANDOM_SEED
from src.supervised.rf_gb_model.cat_boost import prepare_catboost_data, load_data, split_data
from src.supervised.rf_gb_model.cat_boost import LOG_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.supervised.rf_gb_model.model_supervised import OUTPUT

ALL_FEATURES = LOG_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES

def get_oof_residuals(X_train, y_train, train_groups, categorical_columns,
                      best_params, n_splits=5, random_state=RANDOM_SEED):
    """
    Generate out-of-fold CatBoost predictions and residuals.
    Parameters
    X_train : CatBoost-ready training features.
    y_train : Log1p-transformed training targets.
    train_groups : Census tract identifiers aligned with X_train.
    categorical_columns : list of categorical feature names.
    best_params : Hyperparameters selected by GridSearchCV.

    Returns
    DataFrame of OOF predictions and residuals, aligned positionally with the original training data.
    """

    y = np.asarray(y_train).ravel()
    groups = np.asarray(train_groups).ravel()

    if not (len(X_train) == len(y) == len(groups)):
        raise ValueError("Training inputs are not aligned.")

    cv = GroupKFold(n_splits=n_splits)

    oof_preds = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y, groups)):
        model = CatBoostRegressor(**best_params, loss_function="RMSE", random_seed=random_state,
                                  verbose=False, allow_writing_files=False, thread_count=-1)

        model.fit(X_train.iloc[train_idx], y[train_idx], cat_features=categorical_columns)

        oof_preds[val_idx] = model.predict(X_train.iloc[val_idx])

        fold_ids[val_idx] = fold

        print(f"Completed fold {fold + 1}/{n_splits}")

    if np.isnan(oof_preds).any():
        raise RuntimeError("Some OOF predictions are missing.")

    return pd.DataFrame({
        "actual_log": y,
        "pred_log": oof_preds,
        "residual_log": y - oof_preds,
        "fold": fold_ids
        }, index=X_train.index)


def calculate_morans_i(df, residual_col="residual_log", lon_col="longitude", lat_col="latitude",
                       k=8, permutations=999, random_state=RANDOM_SEED):
    """
    Calculate Global Moran's I for spatial residuals.
    Assumes longitude/latitude are in EPSG:4326.

    Returns
    dict of Moran's I, permutation p-value, and diagnostics.
    """

    data = df[[residual_col, lon_col, lat_col]].copy()

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) <= k:
        raise ValueError("Not enough observations for k neighbors.")

    # Convert coordinates to geographic points
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[lon_col], data[lat_col]), crs="EPSG:4326")

    # Project into meters
    gdf = gdf.to_crs("EPSG:5070")
    coords = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])

    # Construct k-nearest-neighbor weights
    w = KNN.from_array(coords, k=k)
    w.transform = "r"

    residuals = gdf[residual_col].to_numpy()

    # Reproducible permutation test
    # Compatible with esda versions using NumPy's global RNG
    old_state = np.random.get_state()
    try:
        np.random.seed(random_state)
        moran = Moran(residuals, w, permutations=permutations)
    finally:
        np.random.set_state(old_state)

    moran_i_dict = {
        "n_retailers": len(gdf),
        "k": k,
        "morans_i": float(moran.I),
        "expected_i": float(moran.EI),
        "p_value": float(moran.p_sim),
        "z_score": float(moran.z_sim)
    }

    return moran_i_dict

def moran_sensitivity(df, k_values=(4, 8, 12), **kwargs):
    results = []

    for k in k_values:
        result = calculate_morans_i(df, k=k, **kwargs)
        results.append(result)

    return pd.DataFrame(results)

def main():

    train_df, _ = load_data()
    X_train, y_train = split_data(train_df)

    X_tr = prepare_catboost_data(X_train, ALL_FEATURES, CATEGORICAL_FEATURES)
    with open(str(OUTPUT / "catboost_metadata.json"), "r") as f:
        best_params = json.load(f)
    tuned_params = {k:best_params[k] for k in {"depth", "iterations", "l2_leaf_reg"} if k in best_params}

    oof_results = get_oof_residuals(X_train=X_tr, y_train=y_train, train_groups=X_train['tract_geoid'],
                                    categorical_columns=CATEGORICAL_FEATURES, best_params=tuned_params)

    spatial_df = oof_results.copy()

    spatial_df["latitude"] = X_train["latitude"].to_numpy()
    spatial_df["longitude"] = X_train["longitude"].to_numpy()
    spatial_df["region"] = X_train["region"].to_numpy()

    nyc_spatial_df = spatial_df[spatial_df['region'] == 'New York City'].copy()

    statewide_result = moran_sensitivity(spatial_df, k_values=(4,8,12))
    nyc_result = moran_sensitivity(nyc_spatial_df, k_values=(4,8,12))

    return statewide_result, nyc_result

if __name__ == '__main__':

    statewide_result, nyc_result = main()

    print("STATEWIDE")
    print(statewide_result)
    print("NYC")
    print(nyc_result)
