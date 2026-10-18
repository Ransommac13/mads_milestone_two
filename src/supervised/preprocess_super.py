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

LOG_FEATURES = ['competitors_within_1_mi', "commercial_places_within_1_mi", "aadt_count", 
                "total_population", "D1C", "rac_c000", 'D1B'] + [f"wac_cns{i:02d}" for i in range (1,21)]


NUMERIC_FEATURES = ['age_18_29_proportion', "age_30_64_proportion", 'poverty_rate',
       'bachelors_degree_proportion_25plus', 'owner_occupied_proportion',
       'family_households_proportion', 'no_vehicle_households_proportion','D3A', 'NatWalkInd', 'wac_ce01_prop', 'wac_ce02_prop',
       'rac_ce01_prop', 'rac_ce02_prop']

CATEGORICAL_FEAUTURES = ["bustype", "chain", "region"]

'''
df[CATEGORICAL_FEAUTURES] = df[CATEGORICAL_FEAUTURES].astype("string")

# Convert all remaining columns to numeric
NUMERIC_FEATURES = df.columns.difference(CATEGORICAL_FEAUTURES)

df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)          
'''

PREPROCESSOR_PATH = Path("supervised/preprocessor.joblib")
CSV_PATH = Path("supervised")

def split_data(df):
    """
    Split dataset into feature and the target (log)
    """
    x = df.drop(columns=[TARGET])
    y = np.log1p(df[TARGET])

    return x, y

def build_preprocessor():
    """
    build preprocessor pipeline for supervised learning
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

    columns = ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_FEAUTURES),
        ("log_numeric", log_pipeline, LOG_FEATURES),
        ("other_numeric_features", numeric_pipeline, NUMERIC_FEATURES)
        ], verbose_feature_names_out=False)

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


    print("Same colimn as train: ")
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

