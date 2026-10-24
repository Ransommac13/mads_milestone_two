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

#print(
#    selected_failures[
#        [
#            "failure_type",
#            "agtno",
#            "actual_sales",
#            "predicted_sales",
#            "absolute_error_log",
#            "absolute_error_dollars",
#        ]
#    ]
#)
#use agtno 112801, 112838, 115618

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.supervised.preprocess_super import split_data
from src.supervised.rf_gb_model.feature_sets import load_split

# Get and print the current working directory

pipeline = joblib.load("src/supervised/outputs/model outputs/random_forest_tuned.joblib")

# Separate preprocessing from the final random forest

preprocessor = pipeline.named_steps["preprocess"]
rf_model = pipeline.named_steps["model"]

#print(type(preprocessor))
#print(type(rf_model))
#print(rf_model.get_params())

val_df = load_split("val")
X_val, y_val = split_data(val_df)

selected_agtno = 112801

failure_matches = failures.loc[
    failures["agtno"] == selected_agtno
]

print(f"Number of matching failure records: {len(failure_matches)}")

failures = val_df.reindex(X_val.index).copy()
failure_index = failure_matches.index[0]

detail_columns = [
    "agtno",
    "actual_sales",
    "predicted_sales",
    "residual_dollars",
    "absolute_error_log",
    "error_direction",
    "sales_category"
]

print(failure_matches[detail_columns])

X_case = X_val.loc[[failure_index]]
print("Selected agtno:", selected_agtno)
print("Validation index:", failure_index)

if "agtno" in X_case.columns:
    print("agtno in X_val:", X_case["agtno"].iloc[0])

X_case_transformed = preprocessor.transform(X_case)

if hasattr(preprocessor, "get_feature_names_out"):
    feature_names = preprocessor.get_feature_names_out()
else:
    feature_names = np.array([
        f"feature_{i}"
        for i in range(X_case_transformed.shape[1])
    ])

if hasattr(X_case_transformed, "toarray"):
    X_case_transformed = X_case_transformed.toarray()

X_case_shap = pd.DataFrame(
    np.asarray(X_case_transformed),
    index=[failure_index],
    columns=feature_names
)
#use agtno 112801, 112838, 115618
# -------------------------------------
# Step 5: Create and verify explanation
# -------------------------------------

explainer = shap.TreeExplainer(rf_model)
case_shap = explainer(X_case_shap)

pipeline_prediction = pipeline.predict(X_case)[0]
forest_prediction = rf_model.predict(X_case_shap)[0]

base_value = np.asarray(
    case_shap[0].base_values
).reshape(-1)[0]

shap_prediction = (
    base_value
    + np.asarray(case_shap[0].values).sum()
)

print(f"\nagtno: {selected_agtno}")
print(f"Pipeline log prediction: {pipeline_prediction:.6f}")
print(f"Forest log prediction:   {forest_prediction:.6f}")
print(f"SHAP reconstruction:     {shap_prediction:.6f}")
print(f"Predicted sales: ${np.expm1(pipeline_prediction):,.2f}")
print(
    f"Actual sales: "
    f"${failure_matches['actual_sales'].iloc[0]:,.2f}"
)

# ----------------------------------------
# Step 6: Plot and find features A, B, C
# ----------------------------------------

shap.plots.waterfall(case_shap[0], max_display=12)
plt.title(f"SHAP explanation for agtno {selected_agtno}")
plt.tight_layout()
plt.show()

top_features = (
    pd.DataFrame({
        "agtno": selected_agtno,
        "feature": feature_names,
        "transformed_value": X_case_shap.iloc[0].to_numpy(),
        "shap_value": np.asarray(case_shap[0].values)
    })
    .assign(
        absolute_shap=lambda df: df["shap_value"].abs(),
        direction=lambda df: np.select(
            [
                df["shap_value"] > 0,
                df["shap_value"] < 0
            ],
            [
                "increased predicted log sales",
                "decreased predicted log sales"
            ],
            default="no effect"
        )
    )
    .sort_values("absolute_shap", ascending=False)
)

top_three = top_features.head(3).copy()
top_three.insert(
    1,
    "rank",
    ["Feature A", "Feature B", "Feature C"]
)

print("\nTop three SHAP features:")
print(top_three.to_string(index=False))

'''
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
'''