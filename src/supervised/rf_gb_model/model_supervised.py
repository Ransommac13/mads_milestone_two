"""
This script will uses random forest and gradient boosting models to predict a retailer's avg daily lottery sales
- Compare feature representations with grouped corss-validation 
- tune models on the best representation 
- evaluate on validation 
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from scipy.stats import spearmanr

from src.config import RANDOM_SEED, TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH
from src.supervised.preprocess_super import PREPROCESSOR_PATH, build_preprocessor, split_data
from src.supervised.rf_gb_model.feature_sets import load_split, get_feature_sets
from src.supervised.rf_gb_model.visualizations_rf_gm import plot_error_by_group, plot_feature_set_comparison, plot_predicted_vs_actual, plot_importance, plot_residuals

OUTPUT = Path("src/supervised/outputs/model outputs")
FIGURE_PATH = Path("src/supervised/outputs/model figures")

# 5-fold cross validation, retailers in the same census tray stay in the same fold
CV = GroupKFold(n_splits=5)

def get_models():
    """
    Random Forest and Gradient Boosting Model

    """

    return {
        #300 trees, at least 15 retailers per ;ear, 
        "random_forest":
        RandomForestRegressor(n_estimators=300, min_samples_leaf=15, max_features=0.5, n_jobs=-1,
                                  random_state=RANDOM_SEED),

        "gradient_boosting": HistGradientBoostingRegressor(learning_rate=0.05, 
                                                           max_depth=4, max_iter=300,
                                                           min_samples_leaf=25, 
                                                           random_state=RANDOM_SEED)
    }

# tuning for finding the best settings for a model

def tunings_settings():
    """
    Settings to try when tuning a random sample of these tests
    """

    return {

        "random_forest": {"model__n_estimators":[200,300],
                          "model__min_samples_leaf": [5,10,15,20,40],
                          "model__max_features": [0.3, 0.5, 0.7, 1.0],
                          "model__max_depth": [None, 10, 20]},
        
        "gradient_boosting": {"model__learning_rate":[0.02,0.05, 0.1],
                          "model__min_samples_leaf": [20,40,60, 80, 150],
                          "model__max_iter": [200, 300, 500],
                          "model__max_depth": [3, 4, 6, None],
                          "model__l2_regularization": [0,1,10]}

    }

def make_pipeline(model, feature_set_kwargs):
    """
    make preprocessor and the model into a single object 
    """
    return Pipeline([("preprocess", build_preprocessor(**feature_set_kwargs)),("model", model)])


def check_model(model, X, y_log):

    """
    use MAE, RMSE, and R3 on the log scale and in dollars
    """
    #predict log sales
    pred_log = model.predict(X)
    # convert both prediction and actual back to dollars 
    pred, actual = np.expm1(pred_log), np.expm1(y_log)

    result = {}

    #scale models are trained on 
    result["MAE (log)"] = mean_absolute_error(y_log, pred_log)
    # average error
    result["RMSE (log)"] = np.sqrt(mean_squared_error(y_log, pred_log))
    result["R2 (log)"] = r2_score(y_log, pred_log)
    result["Spearman"] = spearmanr(y_log, pred_log)[0]

    result["MAE ($)"] = mean_absolute_error(actual, pred)
    result["RMSE ($)"] = np.sqrt(mean_squared_error(actual, pred))
    result["R2 ($)"] = r2_score(actual, pred)

    return result


def compare_feature(X_train, y_train, groups, X_val, y_val):

    rows = []
    for set_name, kwargs in get_feature_sets(X_train).items():
        for model_name, model in get_models().items():
            pipeline = make_pipeline(model, kwargs)

            #group cross-validation on train only data
            cv_scores = cross_val_score(pipeline, X_train, y_train, groups=groups, cv=CV, scoring="r2")

            #fit on all train, then score train to see overfitting and validaiton 
            pipeline.fit(X_train, y_train)
            train_scores = check_model(pipeline, X_train, y_train)
            val_scores = check_model(pipeline, X_val, y_val)
            rows.append({"feature_set": set_name, 
                         "model": model_name,
                         "cv R2 (log)": cv_scores.mean(),
                         "train R2 (log)": train_scores["R2 (log)"],
                         "val R2 (log)": val_scores['R2 (log)'],
                         "val MAE (log)": val_scores["MAE (log)"],
                         "val Spearman": val_scores["Spearman"],
                         "val MAE ($)": val_scores["MAE ($)"]})

            print(f"{set_name:18s} {model_name:18s} cv R2 {cv_scores.mean():.3f} val R2 {val_scores["R2 (log)"]:.3f}")

    return pd.DataFrame(rows)

def tune_model(model_name, feature_set_kwargs, X_train, y_train, groups, n_iter):
    """
    Random search of the model setting 
    """
    pipeline = make_pipeline(get_models()[model_name], feature_set_kwargs)

    search = RandomizedSearchCV(pipeline, tunings_settings()[model_name],
                                n_iter=n_iter, cv=CV, scoring="r2",
                                random_state = RANDOM_SEED, n_jobs=1)

    search.fit(X_train, y_train, groups=groups)

    return search


def feature_importance(pipeline, X, y_log):

    """
    permutation importance on the original columns
    """
    columns = []
    for _,_, cols in pipeline.named_steps["preprocess"].transformers:
        
        columns += list(cols)
    result = permutation_importance(pipeline, X[columns], y_log, n_repeats=5, 
                                    scoring = "neg_mean_absolute_error", 
                                    random_state = RANDOM_SEED)

    return pd.Series(result.importances_mean, index=columns).sort_values(ascending = False)


if __name__ == '__main__':

    # output folders 
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    val_df = load_split("val")

    X_train, y_train = split_data(train_df)
    X_val, y_val = split_data(val_df)
    groups = train_df["tract_geoid"]

    #compare feature representations
    comparison = compare_feature(X_train, y_train, groups, X_val, y_val)
    comparison.round(3).to_csv(OUTPUT / "feature_set_comparison.csv", index=False)
    plot_feature_set_comparison(comparison, FIGURE_PATH / "feature_set_comparison.png")

    #find best representation , highest cross-validation scare on train 
    best_set = comparison.sort_values("cv R2 (log)", ascending=False).iloc[0]["feature_set"]
    best_kwargs = get_feature_sets(X_train)[best_set]
    print(f"\nBest Feature Representation: {best_set}")

    #tune and evaluate 
    rows = []
    for model_name, n_iter in [("random_forest", 10), ("gradient_boosting", 25)]:
        search = tune_model(model_name, best_kwargs, X_train, y_train, groups, n_iter)
        pipeline = search.best_estimator_
        print(f"\n{model_name}: best settings: {search.best_params_}")

        #score train 
        train_scores = check_model(pipeline, X_train, y_train)
        val_scores = check_model(pipeline, X_val, y_val)
        rows.append({"model": model_name, "split": "train", **train_scores})
        rows.append({"model": model_name, "split": "validation", **val_scores})

        rows[-1]["cv R2 (log)"] = search.best_score_
        rows[-1]["best settings"] = str(search.best_params_)

        #save pipeline 
        joblib.dump(pipeline, OUTPUT / f"{model_name}_tuned.joblib")

        pred_val = pd.Series(pipeline.predict(X_val), index=y_val.index)
        title = f"{model_name} validation"
        plot_predicted_vs_actual(y_val, pred_val, title, FIGURE_PATH / f"{model_name}_val_predicted_vs_acutal.png")
        plot_residuals(y_val, pred_val, title, FIGURE_PATH / f"{model_name}_val_residuals.png")
        plot_error_by_group(val_df, y_val, pred_val, "region", f"{title}: error by region", FIGURE_PATH / f"{model_name}_val_error_by_region.png")

        #find features that matter most 
        importance = feature_importance(pipeline, X_val, y_val)
        importance.rename("importance").to_csv(OUTPUT / f"{model_name}_importance.csv")

    results = pd.DataFrame(rows)
    results.drop(columns="best settings").round(3).to_csv(OUTPUT / "tuned_results.csv", index=False)
    results[["model", "best settings"]].dropna().to_csv(OUTPUT / "best_settings.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n Tuned models:")
    print(results.drop(columns=["best settings"]).round(3).to_string(index=False))

    print(f"\nsaved to {OUTPUT}")



        



# results = 
#     X_train_processed = preprocessor.fit_transform(X_train)
#     X_val_processed = preprocessor.transform(X_val)

#     rows = []
#     OUTPUT.mkdir(parents=True, exist_ok=True)

#     for name, model in get_models().items():
#         model.fit(X_train_processed, y_train)
#         joblib.dump(model, OUTPUT / f"{name}.joblib")

#         importance = feature_importance(model, X_val_processed, y_train)

#         print(f"Top 10 features {name}:")
#         print(importance.head(10).round(3).to_string())

#         rows.append({"model": name, "split": "train",
#                     **check_model(model, X_train_processed, y_train)})

#         rows.append({"model": name, "split": "validation",
#                             **check_model(model, X_val_processed, y_train)})

#         results = pd.DataFrame(rows).round(2)
#         results.to_csv(OUTPUT / "validation_results_super.csv", index=False)

#         print(results.to_string(index=False))
    


