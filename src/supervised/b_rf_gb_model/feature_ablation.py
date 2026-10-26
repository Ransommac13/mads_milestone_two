"""
Feature importance (by group) and ablation analysis for the tuned Random Forest and Gradient Boosting models

"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

from src.config import TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH, TEST_DATA_PARQUET_PATH
from src.supervised.a_preprocess_super import LOG_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.supervised.b_rf_gb_model.feature_sets import load_split
from sklearn.inspection import permutation_importance

OUTPUT = Path("src/supervised/outputs/model outputs")
FIGURE_PATH = Path("src/supervised/outputs/model figures")
MODEL_TYPES = ["random_forest", "gradient_boosting"]

CV = GroupKFold(n_splits=5)
N_REPEAT = 10

GROUPS = {
    "Store types": ["bustype", "chain", "region"],
    "Nearby Places": ["competitors_within_1_mi", "commercial_places_within_1_mi", "community_places_within_1_mi"],
    "Traffic": ["aadt_count"],
    "Walkability / density": ["NatWalkInd", 'D1B', 'D1C', 'D3A'],
    "Census": ["total_population", "median_household_income", "poverty_rate", "unemployment_rate","bachelors_degree_proportion_25plus",
               "owner_occupied_proportion", "family_households_proportion", "no_vehicle_households_proportion",
               "age_18_29_proportion", "age_30_64_proportion"],
    "Jobs": ["wac_c000", "wac_ce01_prop", "wac_ce02_prop","rac_c000","rac_ce01_prop","rac_ce02_prop"]

}
ALL_COLUMNS = [c for cols in GROUPS.values() for c in cols]

def build_preprocessor(columns):

    """
    same preprocessor used in preprocess_super but only for the chosen column
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
    for name, pipeline, wanted in [("categorical",categorical_pipeline, CATEGORICAL_FEATURES),
                                   ("log_numeric", log_pipeline, LOG_FEATURES),
                                   ("other_numeric_features", numeric_pipeline, NUMERIC_FEATURES)]:
        
        used = [c for c in wanted if c in columns]
        if used: 
            transformers.append((name, pipeline, used))

    return ColumnTransformer(transformers)

def best_setting(model_type):
    """
    taked the tuned settings from the saved pipeline

    """
    saved_pipeline = joblib.load(OUTPUT / f"{model_type}_tuned.joblib")

    # clone=same settings, no learned trees
    return clone(saved_pipeline.named_steps["model"])

def score_columns(model_type, columns, X_train, y_train, groups, X_val, y_val):
    """
    Train with only these columns, then score with crossvalidation 
    """

    pipeline = Pipeline([("preprocess", build_preprocessor(columns)), ("model", best_setting(model_type))])

    #group cross-validation on train with only selected columns
    cv_scores = cross_val_score(pipeline, X_train[columns], y_train, groups=groups, cv=CV, scoring="r2", n_jobs=1)
    
    #fit on all train, then score train to see overfitting and validaiton 
    pipeline.fit(X_train[columns], y_train)
    pred = pipeline.predict(X_val[columns])

        
    return {"cv R2 (log)": cv_scores.mean(),
           "cv R2 std": cv_scores.std(),
           "val R2 (log)": r2_score(y_val, pred),
            "val MAE ($)": mean_absolute_error(np.expm1(y_val), np.expm1(pred))} #exmp1 converts log sale back into dollars

def group_permutation_importance(model_type, X_val, y_val):
    """
    shuffle all columns of one group together
    """

    pipeline = joblib.load(OUTPUT / f"{model_type}_tuned.joblib")
    # score before any shuffling 
    base_r2 = r2_score(y_val, pipeline.predict(X_val[ALL_COLUMNS]))
    random_state = np.random.RandomState(42)

    rows = []
    for group_name, columns in GROUPS.items():

        drops = []
        for _ in range(N_REPEAT):
            #every column in the group uses the same order
            shuffled = X_val[ALL_COLUMNS].copy()
            order = random_state.permutation(len(shuffled))
            shuffled[columns] = shuffled[columns].values[order]

            drops.append(base_r2 - r2_score(y_val, pipeline.predict(shuffled)))
        # one row per group
        rows.append({"model": model_type, "group": group_name, "features": len(columns),
                         "R2 drop (mean)": np.mean(drops), "R2 drop(std)": np.std(drops)})

    return pd.DataFrame(rows)


def run_ablation(model_type, X_train, y_train, groups, X_val, y_val):
    """
    All ablation experiments for one model
    """
    results = []

    def add(experiment, removed_or_added, columns):
        scores = score_columns(model_type, columns, X_train, y_train, groups, X_val, y_val)
        results.append({"model": model_type, "experiment": experiment, "setting":removed_or_added,
                       "features": len(columns), **scores})

        print(f"{model_type:18s} {experiment:16s} {removed_or_added:28s} val R2 {scores['val R2 (log)']:.3f}", flush=True)

    add("full model", "all feature", ALL_COLUMNS)
    # drop one group 
    for group_name, columns, in GROUPS.items():
        add("drop one group", group_name, [c for c in ALL_COLUMNS if c not in columns])

    #only one group 
    for group_name, columns, in GROUPS.items():
        add("only one group", group_name, columns)

    #adding one group at a time 
    used = []

    for group_name, columns in GROUPS.items():
        used = used + columns 
        add("build up", f" + {group_name}", list(used))

    # drop one store label 
    for label in GROUPS["Store types"]:
        add("drop one label", label, [c for c in ALL_COLUMNS if c!= label])


    return pd.DataFrame(results)

def plot_group_importance(importance, path):

    """
    Bars: how much R2 is lost when a group is shuffled
    """

    # one column per model, one row per group
    table = importance.pivot_table(index="group", columns= "model", values="R2 drop (mean)").sort_values("random_forest")
    table.plot.barh(figsize=(9,5))

    plt.ylabel("")
    plt.xlabel("Drop in validation R2 (LOG) When group is shuffled")
    plt.title("Group permutation importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_drop_one_group(ablation, path):

    """
    Bars: R2 lost when a whole group is removed and model is re-trained
    """
    #R2 of the full model for each model
    full = ablation[ablation["experiment"] == "full model"].set_index("model")["val R2 (log)"]
    #R2 when each group is dropped 
    drop = ablation[ablation["experiment"] == "drop one group"].pivot(index="setting", columns="model", values="val R2 (log)")

    #loss = fulll R2 - R2 without the group
    loss = (full - drop).sort_values("random_forest")
    loss.plot.barh(figsize=(9,5))

    plt.ylabel("")
    plt.xlabel("Drop in validation R2 (LOG) When group is removed")
    plt.title("Ablation: what we lose without each group")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    


def plot_build_up(ablation, path):

    """
    Line : validation R2 as groups are added one at a time

    """
    build = ablation[ablation["experiment"] == "build up"]
    #pivot sorts the rows a-z, .loc puts them back in order the groups were added
    table = build.pivot(index="setting", columns="model", values="val R2 (log)").loc[build["setting"].unique()]

    table.plot(marker="o", figsize=(9,5), rot=25)

    plt.xlabel("")
    plt.ylabel("Validation R2 (log)")
    plt.title("Build Up: adding one group at a time")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()



if __name__ == "__main__":

    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    val_df = load_split("val")
    y_train, y_val = np.log1p(train_df[TARGET]), np.log1p(val_df[TARGET])

    groups=train_df["tract_geoid"]

    # group permutation importance 
    importance = pd.concat([group_permutation_importance(name, val_df, y_val) for name in MODEL_TYPES])
    importance.round(4).to_csv(OUTPUT / "group_importance.csv", index=False)

    plot_group_importance(importance, FIGURE_PATH / "group_importance.png")
    print(importance.round(3).to_string(index=False))

    #ablation
    ablation = pd.concat([run_ablation(name, train_df, y_train, groups, val_df, y_val) for name in MODEL_TYPES])
    ablation.round(3).to_csv(OUTPUT / "ablation_results.csv", index=False)
    plot_drop_one_group(ablation, FIGURE_PATH / "Ablation drop one group.png")
    plot_build_up(ablation, FIGURE_PATH / "ablation build up.png")

    print(f"\nsaved to {OUTPUT} and {FIGURE_PATH}")
   


