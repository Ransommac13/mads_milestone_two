"""
This script will load the modeling data and attaches the unspervised learning outputs
For supervised models to compare feature representaitons 
"""
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


from src.config import TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH, TEST_DATA_PARQUET_PATH
from src.supervised.preprocess_super import PREPROCESSOR_PATH, build_preprocessor, split_data
from src.config import TRAIN_PCA_PARQUET_PATH, VAL_PCA_PARQUET_PATH, TEST_PCA_PARQUET_PATH
from src.config import GMM_PARQUET_DIR

# split name (retailer data, pca componets ,gmm probabilites)
SPLIT_PATHS = {
    "train": (TRAIN_DATA_PARQUET_PATH, TRAIN_PCA_PARQUET_PATH, f"{GMM_PARQUET_DIR}/X_gmm_train.parquet"),
    "val": (VALIDATION_DATA_PARQUET_PATH, VAL_PCA_PARQUET_PATH, f"{GMM_PARQUET_DIR}/X_gmm_val.parquet"), 
    "test": (TEST_DATA_PARQUET_PATH, TEST_PCA_PARQUET_PATH, f"{GMM_PARQUET_DIR}/X_gmm_test.parquet"), 

}

def load_split(split):
    """
    load tran, val, test, and add the pca and gmm columns, matched on agtno
    """

    data_path, pca_path, gmm_path = SPLIT_PATHS[split]

    df = pd.read_parquet(data_path)
    pca = pd.read_parquet(pca_path)
    gmm = pd.read_parquet(gmm_path)

    # leep the pc.. and gmm_bro, no clusters 
    pca_columns = [c for c in pca.columns if c.startswith("PC")]
    
    gmm_columns = [c for c in gmm.columns if c.startswith("gmm_prob_")]

    df = df.merge(pca[["agtno"] + pca_columns], on="agtno", how="left", validate="one_to_one")
    df = df.merge(gmm[["agtno"] + gmm_columns], on="agtno", how="left", validate="one_to_one")

    missing = df[pca_columns + gmm_columns].isna().sum().sum()
    print(f"{split}: {missing} retailers have no PCA/GMM values.")

    return df


def get_feature_sets(df):
    """
    the feature representations to cpmapre
    """
    pca_columns = [c for c in df.columns if c.startswith("PC")]
        
    gmm_columns = [c for c in df.columns if c.startswith("gmm_prob_")]

    return {
        "category_only": dict(use_original=False),
        "numeric_only": dict(use_categorical=False),
        "original": dict(),
        "pca": dict(use_original=False, extra_columns=pca_columns),
        "gmm": dict(use_original=False, extra_columns=gmm_columns),
        "original_pca_gmm": dict(extra_columns=pca_columns+gmm_columns)
    }
   
