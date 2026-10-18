#train_ols.py file

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import TRAIN_DATA_PARQUET_PATH

from src.supervised.preprocess_super import build_preprocessor, split_data

OUTPUT_DIR = Path("src/supervised/outputs/ols")
PREPROCESSOR_PATH = OUTPUT_DIR / "preprocessor.joblib"
MODEL_PATH = OUTPUT_DIR / "ols_model.pickle"


def regression_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def evaluate_train(model, X, y_log):
    """
    Evaluate in-sample predictions on:
    1. the log-sales scale used to fit OLS
    2. the original daily-sales scale
    """
    predicted_log = model.predict(X)

    # Convert log1p values back to the original sales scale.
    actual_sales = np.expm1(y_log)
    predicted_sales = np.maximum(np.expm1(predicted_log), 0)

    log_metrics = regression_metrics(y_log, predicted_log)
    sales_metrics = regression_metrics(actual_sales, predicted_sales)

    metrics = {
        "split": "train",
        "log_r2": log_metrics["r2"],
        "log_mae": log_metrics["mae"],
        "log_rmse": log_metrics["rmse"],
        "sales_r2": sales_metrics["r2"],
        "sales_mae": sales_metrics["mae"],
        "sales_rmse": sales_metrics["rmse"],
    }

    predictions = pd.DataFrame(
        {
            "actual_log_sales": np.asarray(y_log),
            "predicted_log_sales": np.asarray(predicted_log),
            "actual_sales": np.asarray(actual_sales),
            "predicted_sales": np.asarray(predicted_sales),
            "log_residual": np.asarray(y_log) - np.asarray(predicted_log),
            "sales_residual": (
                np.asarray(actual_sales) - np.asarray(predicted_sales)
            ),
        }
    )

    return metrics, predictions


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load only the training data.
    train_df = pd.read_parquet(TRAIN_DATA_PARQUET_PATH)
    X_train, y_train = split_data(train_df)

    # Fit preprocessing using only the training data.
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)

    # Remove the identifier from the predictors.
    X_train_processed = X_train_processed.drop(
        columns=["agtno"],
        errors="ignore",
    )

    # Add the OLS intercept.
    X_train_ols = sm.add_constant(
        X_train_processed,
        has_constant="add",
    )

    # Convert to numeric floating-point values.
    X_train_ols = X_train_ols.astype(float)
    y_train = y_train.astype(float)

    # Check predictors and target before fitting.
    if not np.isfinite(X_train_ols.to_numpy()).all():
        raise ValueError(
            "Training predictors contain NaN or infinite values."
        )

    if not np.isfinite(y_train.to_numpy()).all():
        raise ValueError(
            "Training target contains NaN or infinite values."
        )

    # Fit OLS to log1p(avg_daily_sales).
    model = sm.OLS(
        y_train.to_numpy(),
        X_train_ols,
    ).fit()

    # Evaluate the model on the same training data.
    metrics, predictions = evaluate_train(
        model,
        X_train_ols,
        y_train,
    )

    metrics_df = pd.DataFrame([metrics])

    print(metrics_df.to_string(index=False))
    print("\nOLS summary:")
    print(model.summary())

    # Save metrics and fitted values.
    metrics_df.to_csv(
        OUTPUT_DIR / "train_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR / "train_predictions.csv",
        index=False,
    )

    # Save the regression summary.
    (OUTPUT_DIR / "ols_summary.txt").write_text(
        model.summary().as_text(),
        encoding="utf-8",
    )

    # Save the fitted preprocessor and OLS model.
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    model.save(MODEL_PATH)

    print(f"\nPreprocessor saved to: {PREPROCESSOR_PATH}")
    print(f"OLS model saved to: {MODEL_PATH}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()