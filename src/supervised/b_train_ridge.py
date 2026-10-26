"""
Train and evaluate Ridge regression models for lottery retailer sales prediction.

When executed directly, the module:
    1. Loads the original and unsupervised feature datasets.
    2. Applies the saved supervised preprocessor.
    3. Constructs the selected feature configuration.
    4. Tunes and fits Ridge regression on the training dataset.
    5. Evaluates performance on training, validation, and test datasets.
    6. Calculates cross-validation R² statistics.
    7. Exports model performance summaries and cross-validation results.

Inputs:
    - Training, validation, and test retailer datasets (.parquet)
    - Previously generated PCA scores and GMM probabilities (.parquet)
    - Fitted supervised preprocessor and PCA model (.joblib)

Outputs:
    - Ridge regression performance summaries (.csv)
    - Training and validation evaluation metrics (.csv)
    - Cross-validation fold scores (.csv)
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH, TEST_DATA_PARQUET_PATH
from src.config import TRAIN_PCA_PARQUET_PATH, VAL_PCA_PARQUET_PATH, TEST_PCA_PARQUET_PATH, GMM_PARQUET_DIR
from src.supervised.a_preprocess_super import build_preprocessor

TARGET = 'avg_daily_sales'
OUTPUT_DIR = Path("src/supervised/outputs/ridge")
MODEL_OUTPUT = Path("src/supervised/outputs/model outputs")
PREPROCESSOR_PATH = Path("src/supervised/pre_processing_data/preprocessor.joblib")

def load_data():

    train_df = pd.read_parquet(TRAIN_DATA_PARQUET_PATH)
    val_df = pd.read_parquet(VALIDATION_DATA_PARQUET_PATH)

    # Load in pca features for optional addition to or replacement of original features.
    pca_train = pd.read_parquet(TRAIN_PCA_PARQUET_PATH)
    pca_val = pd.read_parquet(VAL_PCA_PARQUET_PATH)
    pca_train5 = pca_train[[f"PC{i}" for i in range(1,6)] + ['agtno']]
    pca_val5 = pca_val[[f"PC{i}" for i in range(1,6)] + ['agtno']]
    pca_train7 = pca_train[[f"PC{i}" for i in range(1,8)] + ['agtno']]
    pca_val7 = pca_val[[f"PC{i}" for i in range(1,8)] + ['agtno']]

    # Load in gmm features for optional addition to original features.
    gmm_dir = Path(GMM_PARQUET_DIR)
    gmm_train = pd.read_parquet(gmm_dir / "X_gmm_train.parquet")[['agtno', 'gmm_prob_0', 'gmm_prob_1', 'gmm_prob_2', 'gmm_prob_3']]
    gmm_val = pd.read_parquet(gmm_dir / "X_gmm_val.parquet")[['agtno', 'gmm_prob_0', 'gmm_prob_1', 'gmm_prob_2', 'gmm_prob_3']]

    # Merge pca and gmm features for optional replacement of and addition to original features.
    gmm_pca_train5 = pca_train5.merge(gmm_train, on='agtno', how='inner', validate='one_to_one')
    gmm_pca_val5 = pca_val5.merge(gmm_val, on='agtno', how='inner', validate='one_to_one')
    gmm_pca_train7 = pca_train7.merge(gmm_train, on='agtno', how='inner', validate='one_to_one')
    gmm_pca_val7 = pca_val7.merge(gmm_val, on='agtno', how='inner', validate='one_to_one')

    test_df = pd.read_parquet(TEST_DATA_PARQUET_PATH)

    pca_test = pd.read_parquet(TEST_PCA_PARQUET_PATH)
    pca_test5 = pca_test[[f"PC{i}" for i in range(1, 6)] + ["agtno"]]
    pca_test7 = pca_test[[f"PC{i}" for i in range(1, 8)] + ["agtno"]]

    gmm_test = pd.read_parquet(
        gmm_dir / "X_gmm_test.parquet"
    )[["agtno", "gmm_prob_0", "gmm_prob_1", "gmm_prob_2", "gmm_prob_3"]]

    gmm_pca_test5 = pca_test5.merge(
        gmm_test,
        on="agtno",
        how="inner",
        validate="one_to_one",
    )

    gmm_pca_test7 = pca_test7.merge(
        gmm_test,
        on="agtno",
        how="inner",
        validate="one_to_one",
    )

    return (
    train_df,
    val_df,
    test_df,
    pca_train,
    pca_val,
    pca_test,
    pca_train5,
    pca_val5,
    pca_test5,
    pca_train7,
    pca_val7,
    pca_test7,
    gmm_train,
    gmm_val,
    gmm_test,
    gmm_pca_train5,
    gmm_pca_val5,
    gmm_pca_test5,
    gmm_pca_train7,
    gmm_pca_val7,
    gmm_pca_test7)

def split_data(df):
    """Separate raw features and log-transformed target."""
    X = df.drop(columns=[TARGET])
    y = np.log1p(df[TARGET])
    return X, y

def merge_unsupervised_features(X_train: pd.DataFrame, features_df: pd.DataFrame, retailer_ids):

    X_train = X_train.copy()
    X_train.insert(0, 'agtno', np.asarray(retailer_ids))

    X_train = X_train.merge(features_df, on='agtno', how='left', validate='one_to_one')
    X_train = X_train.drop(columns='agtno')

    if X_train.isna().any().any():
        raise ValueError("Missing values detected after merging features.")

    return X_train

def preprocess_features(X_train_raw, X_val_raw, preprocessor, cols_to_drop=None, train_addition=None, val_addition=None):

    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    # Add unsupervised features if specified in train_addition and val_addition
    if train_addition is not None and val_addition is not None:
        X_train = merge_unsupervised_features(X_train, train_addition, X_train_raw["agtno"])
        X_val = merge_unsupervised_features(X_val, val_addition,  X_val_raw["agtno"])

    if cols_to_drop:
        columns = [col for col in cols_to_drop if col in X_train.columns]
        X_train = X_train.drop(columns=columns, errors="ignore")
        X_val = X_val.drop(columns=columns, errors="ignore")

    X_train = X_train.astype(float)
    X_val = X_val.astype(float)

    return X_train, X_val

def make_cv():
    return KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

def fit_ridge(X_train, y_train):
    model = RidgeCV(
        alphas=np.logspace(-4, 4, 100),
        cv=make_cv(),
        scoring="neg_mean_squared_error",
    )

    model.fit(X_train, y_train)
    return model

def calculate_cv_r2(X_train, y_train):
    """
    Calculate held-out R² on the log-transformed target for each fold.

    RidgeCV performs inner CV to choose alpha, while cross_validate performs
    outer CV to estimate held-out performance.
    """
    model = RidgeCV(
        alphas=np.logspace(-4, 4, 100),
        cv=make_cv(),
        scoring="neg_mean_squared_error",
    )

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=make_cv(),
        scoring="r2",
        return_train_score=True,
        n_jobs=-1,
    )

    return {
        "cv_train_r2_log_mean": np.mean(cv_results["train_score"]),
        "cv_train_r2_log_std": np.std(
            cv_results["train_score"], ddof=1
        ),
        "cv_r2_log_mean": np.mean(cv_results["test_score"]),
        "cv_r2_log_std": np.std(
            cv_results["test_score"], ddof=1
        ),
        "cv_r2_log_folds": cv_results["test_score"],
    }

def evaluate(y_val, X_val, model, retailer_ids=None):

    y_true_log = np.asarray(y_val)
    y_true_sales = np.expm1(y_true_log)

    predictions_log = np.asarray(model.predict(X_val))
    predictions_sales = np.maximum(np.expm1(predictions_log), 0)

    metrics = {
        "r_squared_log": r2_score(y_true_log, predictions_log),
        "rmse_log": np.sqrt(mean_squared_error(y_true_log, predictions_log)),
        "mae_log": mean_absolute_error(y_true_log, predictions_log),
        "r_squared_sales": r2_score(y_true_sales, predictions_sales),
        "rmse_sales": np.sqrt(mean_squared_error(y_true_sales, predictions_sales)),
        "mae_sales": mean_absolute_error(y_true_sales, predictions_sales),
    }

    residual_results = {
        "actual_log_sales": y_true_log,
        "predicted_log_sales": predictions_log,
        "log_residual": y_true_log - predictions_log,
        "actual_sales": y_true_sales,
        "predicted_sales": predictions_sales,
        "sales_residual": y_true_sales - predictions_sales,
    }
    
    metrics_df = pd.DataFrame([{
        "model": "Ridge",
        **metrics
    }])
    residuals_df = pd.DataFrame(residual_results)
    
    if retailer_ids is not None:
        residuals_df.insert(0, "agtno", np.asarray(retailer_ids))

    return metrics_df, residuals_df

def preprocess_evaluation_features(
    X_raw,
    preprocessor,
    cols_to_drop=None,
    feature_addition=None,
):
    X = preprocessor.transform(X_raw)

    if feature_addition is not None:
        X = merge_unsupervised_features(
            X,
            feature_addition,
            X_raw["agtno"],
        )

    if cols_to_drop:
        columns = [col for col in cols_to_drop if col in X.columns]
        X = X.drop(columns=columns, errors="ignore")

    return X.astype(float)
  
if __name__ == "__main__":

    # Load raw data

    train_df, val_df, test_df,\
    pca_train, pca_val, pca_test, pca_train5, pca_val5, pca_test5, pca_train7, pca_val7, pca_test7,\
    gmm_train, gmm_val, gmm_test, gmm_pca_train5, gmm_pca_val5, gmm_pca_test5, gmm_pca_train7, gmm_pca_val7, gmm_pca_test7\
    = load_data()

    #Split raw data
    X_train_raw, y_train = split_data(train_df)
    X_val_raw, y_val = split_data(val_df)
    X_test_raw, y_test = split_data(test_df)

    #Load the already-fitted preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Identify cols used to develop PCs for optional replacement.
    pca = joblib.load("src/unsupervised_clustering/fitted_objects/pca.joblib")
    pca_source_features = pca.feature_names_in_.tolist()
    pca_source_features = [feature.removeprefix("log1p_") for feature in pca_source_features]
    
    # Optionally define cols to drop for investigative purposes.
    drop_cols_dict = {
        'region_cols': ['region_Capital Region', 'region_Central New York', 'region_Finger Lakes', 'region_Long Island', 'region_Mid-Hudson',
       'region_Mohawk Valley', 'region_New York City', 'region_North Country', 'region_Southern Tier', 'region_Western New York'],
        'bustype_infrequent': ['bustype_infrequent_sklearn'],
        'pca_cols': pca_source_features
        }
    cols_to_drop = [item for sublist in drop_cols_dict.values() for item in sublist]

    #Apply preprocessor to X_train_raw and X_val_raw
    X_train, X_val = preprocess_features(X_train_raw, X_val_raw, preprocessor, cols_to_drop=cols_to_drop,
                                         train_addition=gmm_pca_train5,
                                         val_addition=gmm_pca_val5)

    X_test = preprocess_evaluation_features(
        X_raw=X_test_raw,
        preprocessor=preprocessor,
        cols_to_drop=cols_to_drop,
        feature_addition=gmm_pca_test5,
    )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and test feature columns do not match."
        )

    # Fit model
    ridge_model = fit_ridge(X_train, y_train)
    print(f"Selected alpha: {ridge_model.alpha_}")

    # Cross-validation log R²
    cv_metrics = calculate_cv_r2(X_train, y_train)
    cv_r2_log = cv_metrics["cv_r2_log_mean"]
    print(
        f"Cross-validation R² log: {cv_r2_log:.6f} "
        f"+/- {cv_metrics['cv_r2_log_std']:.6f}"
    )

    # Evaluate the fitted model on the full training set
    train_metrics_df, train_residuals_df = evaluate(
        y_val=y_train,
        X_val=X_train,
        model=ridge_model,
        retailer_ids=train_df["agtno"].to_numpy(),
    )

    # Evaluate the fitted model on the separate validation set
    val_metrics_df, val_residuals_df = evaluate(
        y_val=y_val,
        X_val=X_val,
        model=ridge_model,
        retailer_ids=val_df["agtno"].to_numpy(),
    )

    test_metrics_df, test_residuals_df = evaluate(
        y_val=y_test,
        X_val=X_test,
        model=ridge_model,
        retailer_ids=test_df["agtno"].to_numpy(),
    )
    test_r2_log = test_metrics_df.loc[0, "r_squared_log"]
    print(f"Test R² log:       {test_r2_log:.6f}")

    # Calculate nested cross-validation R² on the training set
    cv_metrics = calculate_cv_r2(X_train, y_train)

    # Extract the requested log-scale R² values
    train_r2_log = train_metrics_df.loc[0, "r_squared_log"]
    validation_r2_log = val_metrics_df.loc[0, "r_squared_log"]
    cv_r2_log = cv_metrics["cv_r2_log_mean"]

    print(f"Train R² log:      {train_r2_log:.6f}")
    print(f"Validation R² log: {validation_r2_log:.6f}")
    print(
        f"Cross-validation R² log: {cv_r2_log:.6f} "
        f"+/- {cv_metrics['cv_r2_log_std']:.6f}"
    )
    print(f"Cross-validation fold scores: {cv_metrics['cv_r2_log_folds']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    r2_summary_df = pd.DataFrame([{
        "model": "Ridge",
        "selected_alpha": ridge_model.alpha_,
        "train_r2_log": train_r2_log,
        "validation_r2_log": validation_r2_log,
        "cv_train_r2_log_mean": cv_metrics["cv_train_r2_log_mean"],
        "cv_train_r2_log_std": cv_metrics["cv_train_r2_log_std"],
        "cv_r2_log_mean": cv_metrics["cv_r2_log_mean"],
        "cv_r2_log_std": cv_metrics["cv_r2_log_std"],
    }])

    r2_summary_df.to_csv(
        OUTPUT_DIR / "ridge_r2_summary.csv",
        index=False,
    )

    train_metrics_df.to_csv(
        OUTPUT_DIR / "ridge_train_metrics.csv",
        index=False,
    )

    val_metrics_df.to_csv(
        OUTPUT_DIR / "ridge_validation_metrics.csv",
        index=False,
    )

    pd.DataFrame({
        "fold": np.arange(1, len(cv_metrics["cv_r2_log_folds"]) + 1),
        "cv_r2_log": cv_metrics["cv_r2_log_folds"],
    }).to_csv(
        OUTPUT_DIR / "ridge_cv_fold_scores.csv",
        index=False,
    )

    joblib.dump(ridge_model, MODEL_OUTPUT / f"ridge_tuned.joblib")
