#train_ols.py file

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, r2_score

from src.config import TRAIN_DATA_PARQUET_PATH
from src.supervised.preprocess_super import build_preprocessor, split_data

OUTPUT_DIR = Path("src/supervised/outputs/ols")
PREPROCESSOR_PATH = OUTPUT_DIR / "preprocessor.joblib"
MODEL_PATH = OUTPUT_DIR / "ols_model.pickle"
SUMMARY_PATH = OUTPUT_DIR / "ols_summary.txt"
METRICS_PATH = OUTPUT_DIR / "train_metrics.csv"
RESIDUALS_PATH = OUTPUT_DIR / "train_residuals.csv"

def main():
    # Create the output directory.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load only the training data.
    train_df = pd.read_parquet(TRAIN_DATA_PARQUET_PATH)

    # This returns X and log1p(avg_daily_sales).
    X_train, y_train = split_data(train_df)

    # Fit preprocessing using only the training data.
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)

    # Do not use the identifier as a predictor.
    X_train_processed = X_train_processed.drop(
        columns=["agtno", "longitude", "latitude", "bg_geoid", "tract_geoid"],
        errors="ignore",
    )

    # Add the intercept required by statsmodels OLS.
    X_train_ols = sm.add_constant(
        X_train_processed,
        has_constant="add",
    )

    # Convert all model inputs to floating-point values.
    X_train_ols = X_train_ols.astype(float)
    y_train = y_train.astype(float)

    # Check for values OLS cannot process.
    if not np.isfinite(X_train_ols.to_numpy()).all():
        raise ValueError(
            "Training predictors contain missing or infinite values. "
            "Check the preprocessing pipeline."
        )

    if not np.isfinite(y_train.to_numpy()).all():
        raise ValueError(
            "The training target contains missing or infinite values."
        )

    # Fit OLS to log1p(avg_daily_sales).
    model = sm.OLS(
        y_train.to_numpy(),
        X_train_ols,
    ).fit()

    # In-sample predictions and residuals.
    predicted_log_sales = model.predict(X_train_ols)
    residuals = y_train.to_numpy() - np.asarray(predicted_log_sales)

    # Training metrics on the log scale.
    r_squared = r2_score(y_train, predicted_log_sales)
    rmse = np.sqrt(mean_squared_error(y_train, predicted_log_sales))
    rmse = np.exp(rmse)

    print(f"Training R-squared: {r_squared:.4f}")
    print(f"Training RMSE:      {rmse:.4f}")
    print("\nOLS summary:")
    print(model.summary())

    # Save R-squared and RMSE.
    metrics = pd.DataFrame(
        {
            "r_squared": [r_squared],
            "rmse": [rmse],
        }
    )
    metrics.to_csv(METRICS_PATH, index=False)

    # Save observations, fitted values, and residuals.
    residual_results = pd.DataFrame(
        {
            "actual_log_sales": y_train.to_numpy(),
            "predicted_log_sales": np.asarray(predicted_log_sales),
            "residual": residuals,
        }
    )

    residual_results.to_csv(
        RESIDUALS_PATH,
        index=False,
    )

    # Save the OLS summary.
    SUMMARY_PATH.write_text(
        model.summary().as_text(),
        encoding="utf-8",
    )

    # Save the fitted preprocessor and model.
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()