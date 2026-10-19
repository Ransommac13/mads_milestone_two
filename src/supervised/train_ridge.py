"""

train_ridge.py

Ridge! Ridge! Ridge!
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold

from src.config import TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH, TRAIN_PCA_PARQUET_PATH, VAL_PCA_PARQUET_PATH, GMM_PARQUET_DIR
from src.supervised.preprocess_super import build_preprocessor

TARGET = 'avg_daily_sales'

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

