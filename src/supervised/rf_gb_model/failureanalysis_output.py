'''
Import results of failure analysis and select three records for the results
'''
import pandas as pd

failures = pd.read_csv("src/supervised/outputs/model outputs/random_forest_failures.csv")
print(failures.head(10))

rf_failures = (failures.sort_values("absolute_error_log", ascending=False))

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
            "actual_sales",
            "predicted_sales",
            "absolute_error_log",
            "absolute_error_dollars",
        ]
    ]
)