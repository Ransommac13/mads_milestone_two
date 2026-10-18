"""

hdbscan.py
This module does some magic.
"""

import numpy as np
import pandas as pd
from itertools import product
import hdbscan

from src.config import TRAIN_PCA_PARQUET_PATH

def explore_hdbscan_parameter_selections(X_train: pd.DataFrame, min_cluster_sizes: list, min_samples_values: list,
                                         cluster_selection_method='eom', metric='euclidean'):

    """
    Fit HDBSCAN over a set of parameter selections and summarize cluster structure.

    Parameters
    X_train : DataFrame of numeric features. Excludes agtno.
    min_cluster_sizes : list of values of min_cluster_size to explore.
    min_samples_values : list of values of min_samples to explore.
    cluster_selection_method : "eom" or "leaf".
    metric : Distance metric.

    Returns
    results_df : DataFrame of summary metrics for each set of parameter selections.
    fitted_models : dictionary of fitted models. Keys are (min_cluster_size, min_samples).
    """

    results = []
    fitted_models = {}

    for mcs, ms in product(min_cluster_sizes, min_samples_values):

        model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
            cluster_selection_method=cluster_selection_method, metric=metric, prediction_data=True)

        model.fit(X_train)

        labels = model.labels_
        clustered_true = labels != -1

        cluster_sizes = (pd.Series(labels[clustered_true]).value_counts())
        n_clusters = len(cluster_sizes)
        noise_pct = (~clustered_true).mean() * 100

        results.append({
            "min_cluster_size": mcs,
            "min_samples": ms,
            "n_clusters": n_clusters,
            "noise_pct": noise_pct,
            "largest_cluster_pct": (cluster_sizes.max() / len(labels) * 100 if n_clusters > 0 else np.nan),
            "smallest_cluster_size": (cluster_sizes.min() if n_clusters > 0 else np.nan),
            "median_cluster_size": (cluster_sizes.median() if n_clusters > 0 else np.nan),
            "mean_membership_strength": (model.probabilities_[clustered_true].mean() if clustered_true.any() else np.nan),
            "mean_cluster_persistence": (model.cluster_persistence_.mean() if n_clusters > 0 else np.nan)
        })

        fitted_models[(mcs, ms)] = model

    results_df = pd.DataFrame(results).sort_values(["min_cluster_size", "min_samples"]).reset_index(drop=True)

    return results_df, fitted_models



if __name__ == '__main__':

    X_pca_train = pd.read_parquet(TRAIN_PCA_PARQUET_PATH)
    train_pca5 = X_pca_train[[f"PC{i}" for i in range (1,6)]]
    train_pca3 = X_pca_train[[f"PC{i}" for i in range (1,4)]]
    train_pca2 = X_pca_train[[f"PC{i}" for i in range (1,3)]]

    X_train_processed = pd.read_parquet("src/unsupervised_clustering/processed_data/X_train_processed.parquet")
    
    initial_results_df, initial_models_dict = explore_hdbscan_parameter_selections(
        X_train=X_train_processed,
        min_cluster_sizes=[20, 40, 80],
        min_samples_values=[3, 5, 10]
    )

    print("Original X Train Features")
    print(initial_results_df.round(3))
