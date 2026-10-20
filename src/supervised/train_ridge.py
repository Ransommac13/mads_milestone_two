"""

train_ridge.py

Ridge! Ridge! Ridge!
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH, TRAIN_PCA_PARQUET_PATH, VAL_PCA_PARQUET_PATH, GMM_PARQUET_DIR
from src.supervised.preprocess_super import build_preprocessor

TARGET = 'avg_daily_sales'
OUTPUT_DIR = Path("src/supervised/outputs/ridge")
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

    return train_df, val_df, pca_train, pca_val, pca_train5, pca_val5, pca_train7, pca_val7, gmm_train, gmm_val, gmm_pca_train5, gmm_pca_val5, gmm_pca_train7, gmm_pca_val7

def split_data(df):
    """Separate raw features and log-transformed target."""
    X = df.drop(columns=[TARGET])
    y = np.log1p(df[TARGET])
    return X, y

#####
def merge_unsupervised_features(X_train: pd.DataFrame, features_df: pd.DataFrame, retailer_ids):

    X_train = X_train.copy()
    X_train.insert(0, 'agtno', np.asarray(retailer_ids))

    X_train = X_train.merge(features_df, on='agtno', how='left', validate='one_to_one')
    X_train = X_train.drop(columns='agtno')

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


def fit_ridge(X_train, y_train):
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    model = make_pipeline(
        StandardScaler(),
        RidgeCV(
            alphas=np.logspace(-4, 4, 100),
            cv=cv,
            scoring="neg_mean_squared_error",
        ),
    )

    model.fit(X_train, y_train)
    return model

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
        "model": "OLS",
        **metrics
    }])
    residuals_df = pd.DataFrame(residual_results)
    
    if retailer_ids is not None:
        residuals_df.insert(0, "agtno", np.asarray(retailer_ids))

    return metrics_df, residuals_df


  
if __name__ == "__main__":

    # Load raw data
    train_df, val_df,\
    pca_train, pca_val, pca_train5, pca_val5, pca_train7, pca_val7,\
    gmm_train, gmm_val, gmm_pca_train5, gmm_pca_val5, gmm_pca_train7, gmm_pca_val7\
    = load_data()

    #Split raw data
    X_train_raw, y_train = split_data(train_df)
    X_val_raw, y_val = split_data(val_df)

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

    # Fit OLS on training data
    ridge_model = fit_ridge(X_train, y_train)
    print(ridge_model.summary())

    train_metrics_df, train_residuals_df = evaluate(y_val=y_train, X_val=X_train, model=ols_model, retailer_ids=train_df['agtno'].to_numpy())
    val_metrics_df, val_residuals_df = evaluate(y_val=y_val, X_val=X_val, model=ols_model, retailer_ids=val_df['agtno'].to_numpy())

    print("TRAIN", train_metrics_df)
    print("VAL", val_metrics_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Save the OLS summary and model
    #SUMMARY_PATH.write_text(ols_model.summary().as_text(), encoding="utf-8")
    #ols_model.save(MODEL_PATH)