"""

categorical_baseline.py

This module will solely use a simple regression to determine how much variance in avg_daily_sales can be captured solely by knowing
a retailer's most important categorial variables (bustype, chain, and region)
"""


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

from src.config import TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH

TARGET = 'avg_daily_sales'

def split_data(df):
    """Separate raw features and log-transformed target."""
    X = df.drop(columns=[TARGET])
    y = np.log1p(df[TARGET])
    return X, y


def load_data():

    train_df = pd.read_parquet(TRAIN_DATA_PARQUET_PATH)
    val_df = pd.read_parquet(VALIDATION_DATA_PARQUET_PATH)

    return train_df, val_df

class SmoothedCategoryRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, columns=("bustype",), smoothing=20):
        self.columns = columns
        self.smoothing = smoothing

    def fit(self, X, y):
        X = X.loc[:, list(self.columns)].copy().reset_index(drop=True)
        X = X.fillna("__MISSING__").astype(str)

        y = np.asarray(y, dtype=float).ravel()

        if len(X) != len(y):
            raise ValueError("X and y must have equal lengths.")

        self.global_mean_ = float(np.mean(y))

        df = X.copy()
        df["_target"] = y

        stats = df.groupby(list(self.columns), dropna=False)["_target"].agg(["mean", "count"]).reset_index()

        m = self.smoothing

        stats["_prediction"] = ((stats["count"] * stats["mean"] + m * self.global_mean_) / (stats["count"] + m))

        self.lookup_ = stats[list(self.columns) + ["_prediction"]]

        return self

    def predict(self, X):
        X = X.loc[:, list(self.columns)].copy().reset_index(drop=True)
        X = X.fillna("__MISSING__").astype(str)

        merged = X.merge(self.lookup_, on=list(self.columns), how="left", sort=False, validate="many_to_one")

        return (merged["_prediction"].fillna(self.global_mean_).to_numpy())

def fit_and_evaluate_model(X_train, y_train, X_val, y_val):

    
    # Existing training and validation data:
    # X_train, X_val, y_train, y_val
    # train_groups: block group IDs aligned with X_train

    feature_sets = {
        "bustype": ("bustype",),
        "bustype_chain": ("bustype", "chain"),
        "bustype_chain_region": ("bustype", "chain", "region")
    }

    cv = GroupKFold(n_splits=5)
    smoothing_grid = [0, 5, 20, 50, 100, 200]

    results = []
    fitted_models = {}

    for name, columns in feature_sets.items():

        model = SmoothedCategoryRegressor(columns=columns)

        search = GridSearchCV(
            estimator=model,
            param_grid={"smoothing": smoothing_grid},
            cv=cv,
            scoring="r2",
            refit=True,
            n_jobs=-1
        )

        search.fit(
            X_train,
            y_train,
            groups=X_train['tract_geoid']
        )

        best_model = search.best_estimator_
        fitted_models[name] = best_model

        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_val)

        results.append({
            "feature_set": name,
            "best_smoothing": search.best_params_["smoothing"],
            "cv_r2_log": search.best_score_,
            "train_r2_log": r2_score(y_train, train_pred),
            "val_r2_log": r2_score(y_val, val_pred),
            "val_mae_log": mean_absolute_error(y_val, val_pred)
        })

    results_df = pd.DataFrame(results)

    return results_df, fitted_models

def main():

    X_train, X_val = load_data()
    X_train, y_train = split_data(X_train)
    X_val, y_val = split_data(X_val)

    results, best_cat_model = fit_and_evaluate_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    return results, best_cat_model

if __name__ == '__main__':

    results, best_cat_model = main()

    print(results)

