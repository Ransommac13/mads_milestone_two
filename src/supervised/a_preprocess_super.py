"""
Preprocess retailer features and prepare datasets for supervised sales prediction.

This module prepares retailer-level features and target variables for regression
models predicting average daily New York Lottery sales.
It provides a configurable preprocessing pipeline supporting original numerical features, categorical features,
and additional engineered features, specifically PCA scores and GMM membership probabilities.

Preprocessing includes:
    1. Separating predictor variables from average daily sales and applying
       a log1p transformation to the target.
    2. Median-imputing missing numerical values.
    3. Applying log1p transformations to selected skewed numerical features.
    4. Standardizing numerical features using StandardScaler.
    5. Imputing missing categorical values using the most frequent category.
    6. One-hot encoding categorical variables, grouping infrequent categories
       and handling previously unseen categories.

The build_preprocessor() function supports alternative feature configurations,
allowing downstream models to compare categorical features, original numerical
features, and optional PCA/GMM-derived features.

When executed directly, the module:
    1. Loads the training, validation, and test datasets.
    2. Separates predictors and log-transformed sales targets.
    3. Fits the default preprocessing pipeline exclusively on training data.
    4. Transforms validation and test data using the fitted pipeline.
    5. Checks for remaining missing values and consistent feature columns.
    6. Saves the fitted preprocessor as a joblib file.
    7. Exports processed datasets, retaining retailer identifiers (agtno)
       and log-transformed sales targets, as CSV files.

The training-only fitting procedure prevents information leakage from
validation and test datasets. Retailer identifiers are preserved in exported
datasets for downstream merging and analysis but are not used as predictors.

Inputs:
    - Training, validation, and test retailer datasets (.parquet)

Outputs:
    - Fitted supervised preprocessing pipeline (.joblib)
    - Processed training, validation, and test datasets (.csv)
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

from src.config import TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH, TEST_DATA_PARQUET_PATH

TARGET = "avg_daily_sales"

LOG_FEATURES = ["competitors_within_1_mi", "commercial_places_within_1_mi", "community_places_within_1_mi", "aadt_count", 
                "total_population", "D1C", "rac_c000", "D1B", "wac_c000"]

NUMERIC_FEATURES = ["age_18_29_proportion", "age_30_64_proportion", "poverty_rate",
       "bachelors_degree_proportion_25plus", "owner_occupied_proportion", "unemployment_rate",
       "family_households_proportion", "no_vehicle_households_proportion","D3A", "NatWalkInd", "wac_ce01_prop", "wac_ce02_prop",
       "rac_ce01_prop", "rac_ce02_prop", "median_household_income"]

CATEGORICAL_FEATURES = ["bustype", "chain", "region"]


PREPROCESSOR_PATH = Path("src/supervised/pre_processing_data/preprocessor.joblib")
CSV_PATH = Path("src/supervised/pre_processing_data")

def split_data(df):
    """
    Split dataset into feature and the target (log)
    """
    x = df.drop(columns=[TARGET])
    y = np.log1p(df[TARGET])

    return x, y

# def build_preprocessor():
#     """
#     build preprocessor pipeline for supervised learning
#     """
#     categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
#                                      ("onehot", OneHotEncoder(min_frequency=30,
#                                                               handle_unknown="infrequent_if_exist",
#                                                               sparse_output=False))])

#     # skewed numeric columns + log transform
#     log_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
#                             ('log_transform', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
#                             ("scaler", StandardScaler())])

#     # other numeric columns 
#     numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
#                                  ("scaler", StandardScaler())])

#     columns = ColumnTransformer([
#         ("categorical", categorical_pipeline, CATEGORICAL_FEAUTURES),
#         ("log_numeric", log_pipeline, LOG_FEATURES),
#         ("other_numeric_features", numeric_pipeline, NUMERIC_FEATURES)
#         ], verbose_feature_names_out=False)

#     columns.set_output(transform="pandas")

#     return columns
def build_preprocessor(use_original=True, use_categorical=True, extra_columns=None):
    """
    build preprocessor pipeline for supervised learning
    default: original dataset
    use_original: the log + other numeric columns
    use_categorical: bustype , chain, region 
    extra_columns : with the PCA or GMM components
    
    """
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                     ("onehot", OneHotEncoder(min_frequency=30,
                                                              handle_unknown="infrequent_if_exist",
                                                              sparse_output=False))])

    # skewed numeric columns + log transform
    log_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                            ('log_transform', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                            ("scaler", StandardScaler())])

    # other numeric columns 
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler", StandardScaler())])

    transformers = []
    if use_categorical:
        transformers.append(("categorical", categorical_pipeline, CATEGORICAL_FEATURES))
    if use_original:
        transformers.append(("log_numeric", log_pipeline, LOG_FEATURES))
        transformers.append(("other_numeric_features", numeric_pipeline, NUMERIC_FEATURES))

    if extra_columns:
        transformers.append(("extra features", numeric_pipeline, extra_columns))

    columns = ColumnTransformer(
        transformers,verbose_feature_names_out=False)

    # columns = ColumnTransformer([
    #     ("categorical", categorical_pipeline, CATEGORICAL_FEAUTURES),
    #     ("log_numeric", log_pipeline, LOG_FEATURES),
    #     ("other_numeric_features", numeric_pipeline, NUMERIC_FEATURES)
    #     ], verbose_feature_names_out=False)

    columns.set_output(transform="pandas")

    return columns

if __name__ == '__main__':

    X_train, y_train = split_data(pd.read_parquet(TRAIN_DATA_PARQUET_PATH))
    X_val, y_val = split_data(pd.read_parquet(VALIDATION_DATA_PARQUET_PATH))
    X_test, y_test = split_data(pd.read_parquet(TEST_DATA_PARQUET_PATH))

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Check data 
    print("Missing values after preprocssing: ")
    print(" train:", X_train_processed.isna().sum().sum())
    print(" validation:", X_val_processed.isna().sum().sum())
    print(" Test:", X_test_processed.isna().sum().sum())


    print("Same column as train: ")
    print(" validation: ", list(X_val_processed.columns) ==
          list(X_train_processed.columns))
    print(" test: ", list(X_test_processed.columns) ==
              list(X_train_processed.columns))
   
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)


    for name, X_raw, X_processed, y in [("train", X_train, X_train_processed, y_train),
                                         ("val", X_val, X_val_processed, y_val),
                                         ("test", X_test, X_test_processed, y_test)]:
        processed = X_processed.drop(columns=["agtno"], errors="ignore")
        processed.insert(0, "agtno", X_raw["agtno"].values)
        processed["log_avg_daily_sales"] = y.values
        processed.to_csv(CSV_PATH / f"{name}_processed.csv", index=False)

    print(f"Raw columns: {X_train.shape[1]}. Processed features: {X_train_processed.shape[1]}")
    print(f"Train shape: {X_train_processed.shape}. Val shape:{X_val_processed.shape}. Test shape:{X_test_processed.shape}")
    print(f"preprocessor saved to {PREPROCESSOR_PATH}")
    print(f"saved as csv to {CSV_PATH}")

