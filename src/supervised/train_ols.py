'''
train_ols.py file

1 - Fit preprocessing only on training data.
2 - Transform training and validation data using that fitted preprocessor.
3 - Fit OLS only on the transformed training data.
4 - Predict and evaluate on validation data.
5 - Use validation results to make model-development choices.
6 - After finalizing all choices, evaluate once on the untouched test data.
'''

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

from src.config import TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH, TEST_DATA_PARQUET_PATH

TARGET = "avg_daily_sales"
OUTPUT_DIR = Path("src/supervised/outputs/ols")
PREPROCESSOR_PATH = Path("src/supervised/pre_processing_data/preprocessor.joblib")

MODEL_PATH = OUTPUT_DIR / "ols_model.joblib"
SUMMARY_PATH = OUTPUT_DIR / "ols_summary.txt"
METRICS_PATH = OUTPUT_DIR / "train_metrics.csv"
RESIDUALS_PATH = OUTPUT_DIR / "train_residuals.csv"

def split_data(df):
    """Separate raw features and log-transformed target."""
    X = df.drop(columns=[TARGET])
    y = np.log1p(df[TARGET])
    return X, y

def evaluate(y_true_log, y_true_sales, pred_log, pred_sales):
    """Evaluate validation predictions on log and original scales."""

    metrics = {
        "r_squared_log": r2_score(y_true_log, pred_log),
        "rmse_log": mean_squared_error(y_true_log, pred_log) ** 0.5,
        "r_squared_sales": r2_score(y_true_sales, pred_sales),
        "rmse_sales": mean_squared_error(y_true_sales, pred_sales) ** 0.5,
    }

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

    residual_results = pd.DataFrame({
        "actual_log_sales": np.asarray(y_true_log),
        "predicted_log_sales": np.asarray(pred_log),
        "log_residual": np.asarray(y_true_log) - np.asarray(pred_log),
        "actual_sales": np.asarray(y_true_sales),
        "predicted_sales": np.asarray(pred_sales),
        "sales_residual": np.asarray(y_true_sales) - np.asarray(pred_sales),
    })
    residual_results.to_csv(RESIDUALS_PATH, index=False)
  
if __name__ == "__main__":

    #Load raw data.
    train_df = pd.read_parquet(TRAIN_DATA_PARQUET_PATH)
    val_df = pd.read_parquet(VALIDATION_DATA_PARQUET_PATH)

    #X_train_pca = pd.read_parquet('data/modeling/pca/X_pca_train.parquet')
    #X_val_pca = pd.read_parquet('data/modeling/pca/X_pca_val.parquet')

    #Split raw data
    X_train_raw, y_train = split_data(train_df)
    X_val_raw, y_val = split_data(val_df)

    #Load the already-fitted preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    #Apply preprocessor to X_train_raw and X_val_raw
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    #Remove columns to not use as predictors
    X_train = X_train.drop(columns=["agtno", "longitude", "latitude", "bg_geoid", "tract_geoid"],errors="ignore")
    X_val = X_val.drop(columns=["agtno", "longitude", "latitude", "bg_geoid", "tract_geoid"],errors="ignore")

    ######################features to find best model################
    #columns_to_keep = ['competitors_within_1_mi',
    #   'aadt_count','median_household_income', 'total_population', 'poverty_rate','unemployment_rate', 
    #   'family_households_proportion', 'NatWalkInd', 'commercial_places_within_1_mi', 'rac_c000', 'rac_ce01_prop', 'wac_c000']

    #X_train = X_train[columns_to_keep]
    #X_val = X_val[columns_to_keep]

    #X_train = X_train.drop(columns=['agtno', 'longitude', 'latitude', 'bg_geoid', 'tract_geoid', 
    #                                'commercial_places_within_1_mi', 'community_places_within_1_mi', 'wac_cns12', 
    #                                'wac_c000', 'wac_cns19', 'competitors_within_1_mi', 'no_vehicle_households_proportion', 
    #                                'total_population', 'rac_c000'],errors="ignore")
    #X_val = X_val.drop(columns=['agtno', 'longitude', 'latitude', 'bg_geoid', 'tract_geoid', 
    #                                'commercial_places_within_1_mi', 'community_places_within_1_mi', 'wac_cns12', 
    #                                'wac_c000', 'wac_cns19', 'competitors_within_1_mi', 'no_vehicle_households_proportion', 
    #                                'total_population', 'rac_c000'],errors="ignore")

    #Add intercept
    X_train = sm.add_constant(X_train, has_constant="add")
    X_val = sm.add_constant(X_val, has_constant="add")

    # Ensure all columns are numeric
    X_train = X_train.astype(float)
    X_val = X_val.astype(float)

    # Fit OLS on training data
    ols_model = sm.OLS(y_train, X_train).fit()
    print(ols_model.summary())

    # Save the OLS summary and model
    SUMMARY_PATH.write_text(ols_model.summary().as_text(), encoding="utf-8")
    ols_model.save(MODEL_PATH)

    #Evaluate using validation data and convert back to original sales scale
    y_val_sales = val_df[TARGET].to_numpy()

    val_pred_log = np.asarray(ols_model.predict(X_val))
    val_pred_sales = np.maximum(np.expm1(val_pred_log), 0)

    evaluate(
        y_true_log=y_val.to_numpy(),
        y_true_sales=y_val_sales,
        pred_log=val_pred_log,
        pred_sales=val_pred_sales,
    )

