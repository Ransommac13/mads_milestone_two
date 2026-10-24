'''
Import results of failure analysis and select three records for the results
'''
import pandas as pd

failures = pd.read_csv("src/supervised/outputs/model outputs/random_forest_failures.csv")
print(failures.head(10))

rf_failures = (failures.sort_values("absolute_error_log", ascending=False))
print(rf_failures.head(20))


rf_failures["failure_type"] = (
    rf_failures["error_direction"].astype(str)
    + " | "
    + rf_failures["sales_category"].astype(str)
)

# Keep the first—and therefore worst—record of each failure type
selected_failures = rf_failures.drop_duplicates(
    subset="failure_type",
    keep="first"
)

print(
    selected_failures[
        [
            "failure_type",
            "agtno",
            "actual_sales",
            "predicted_sales",
            "absolute_error_log",
            "absolute_error_dollars",
        ]
    ]
)
#use agtno 112801, 112838, 115618

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the fitted pipeline
pipeline = joblib.load("tuned_random_forest.joblib")

# Separate preprocessing from the final random forest
preprocessor = pipeline[:-1]
rf_model = pipeline[-1]

# Select a failure using its index label
failure_id = failures.index[0]
X_case = X_val.loc[[failure_id]]

# Apply fitted preprocessing
X_case_transformed = preprocessor.transform(X_case)

if hasattr(preprocessor, "get_feature_names_out"):
    feature_names = preprocessor.get_feature_names_out()
else:
    feature_names = [
        f"feature_{i}"
        for i in range(X_case_transformed.shape[1])
    ]

if hasattr(X_case_transformed, "toarray"):
    X_case_transformed = X_case_transformed.toarray()

X_case_shap = pd.DataFrame(
    np.asarray(X_case_transformed),
    index=[failure_id],
    columns=feature_names
)

# Explain the random forest's prediction
explainer = shap.TreeExplainer(rf_model)
case_shap = explainer(X_case_shap)

# Plot local feature contributions
shap.plots.waterfall(case_shap[0], max_display=12)
plt.show()

# Extract the three largest contributions
top_features = (
    pd.DataFrame({
        "feature": feature_names,
        "feature_value": X_case_shap.iloc[0].values,
        "shap_value": case_shap[0].values
    })
    .assign(
        absolute_shap=lambda d: d["shap_value"].abs(),
        direction=lambda d: np.where(
            d["shap_value"] >= 0,
            "increased prediction",
            "decreased prediction"
        )
    )
    .sort_values("absolute_shap", ascending=False)
)

print(top_features.head(3))