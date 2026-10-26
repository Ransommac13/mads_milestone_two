"""

final_rf_gb_model_test.py

This module is exclusively used to test tuned Random Forest and Graident Boosting models against the test set,
output performance metrics, and produce explanatory visualizations.
"""

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from src.supervised.a_preprocess_super import split_data
from src.supervised.b_rf_gb_model.feature_sets import load_split
from src.supervised.b_rf_gb_model.b1_model_supervised import check_model, OUTPUT, FIGURE_PATH
from src.supervised.b_rf_gb_model.visualizations_rf_gm import plot_predicted_vs_actual, plot_residuals, plot_error_by_group

if __name__ == '__main__':

    # load test data (with pca and gmm)
    test_df = load_split("test")
    X_test, y_test = split_data(test_df)

    rows=[]
    for model_name in ["random_forest", "gradient_boosting"]:

        pipeline = joblib.load(OUTPUT / f"{model_name}_tuned.joblib")

        rows.append({"model": model_name, "split": "test", **check_model(pipeline, X_test, y_test)})
        
        pred_test = pd.Series(pipeline.predict(X_test), index=y_test.index)
        title = f"{model_name} (test)"
        plot_predicted_vs_actual(y_test, pred_test, title, FIGURE_PATH / f"{model_name}_test_predicted_vs_acutal.png")
        plot_residuals(y_test, pred_test, title, FIGURE_PATH / f"{model_name}_test_residuals.png")
        plot_error_by_group(test_df, y_test, pred_test, "bustype", f"{title}: error by business type", FIGURE_PATH / f"{model_name}_test_error_by_bustype.png")

    #save and print the final result 
    results = pd.DataFrame(rows).round(2)
    results.to_csv(OUTPUT / "test_results.csv", index=False)

    pd.set_option("display.width", 200)
    print(results.to_string(index=False))
        



