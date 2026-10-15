"""

clustering.py

This module makes Reeses's peanut butter clusters (gross!).
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from src.config import RANDOM_SEED
from src.unsupervised_clustering.helper_functions import pc_v_pc_scatterplot

def evaluate_kmeans(X, k_range=range(2,13)):
    results = []

    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)

        labels = model.fit_predict(X)

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X, labels, sample_size=min(3000, len(X)), random_state=RANDOM_SEED)
        })

    return pd.DataFrame(results)

def evaluate_gmm(X_train, X_val, k_range=range(1,13)):
    
    results = []
    for k in k_range:
        for covariance_type in ["full", "diag", "tied"]:
            model = GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=5, reg_covar=1e-5, random_state=RANDOM_SEED)
            model.fit(X_train)

            results.append({
                "n_components": k,
                "covariance": covariance_type,
                "bic": model.bic(X_train),
                "val_log_likelihood": model.score(X_val),
                "converged": model.converged_
            })

    return pd.DataFrame(results)

if __name__ == '__main__':

    X_train_processed = pd.read_parquet("src/unsupervised_clustering/eda_data/X_train_processed.parquet")
    X_val_processed = pd.read_parquet("src/unsupervised_clustering/eda_data/X_val_processed.parquet")

    X_pca_train = pd.read_parquet("src/unsupervised_clustering/eda_data/X_pca_train.parquet").drop(columns='agtno')
    X_pca_val = pd.read_parquet("src/unsupervised_clustering/eda_data/X_pca_val.parquet").drop(columns='agtno')

    train_pca5 = X_pca_train[[f"PC{i}" for i in range(1,6)]]
    val_pca5 = X_pca_val[[f"PC{i}" for i in range(1,6)]]
    train_pca7 = X_pca_train[[f"PC{i}" for i in range(1,8)]]
    val_pca7 = X_pca_val[[f"PC{i}" for i in range(1,8)]]
    
    #results_kmeans_original = evaluate_kmeans(X_train_processed)
    #results_kmeans_five_pcs = evaluate_kmeans(train_pca5)
    #results_kmeans_seven_pcs = evaluate_kmeans(train_pca7)

    results_gmm_original = evaluate_gmm(X_train_processed, X_val_processed).sort_values("bic")
    results_gmm_five_pcs = evaluate_gmm(train_pca5, val_pca5).sort_values("bic")
    results_gmm_seven_pcs = evaluate_gmm(train_pca7, val_pca7).sort_values("bic")

    results_gmm_original.to_csv("src/unsupervised_clustering/eda_data/gmm_results_original.csv")
    results_gmm_five_pcs.to_csv("src/unsupervised_clustering/eda_data/gmm_results_five_pcs.csv")
    results_gmm_seven_pcs.to_csv("src/unsupervised_clustering/eda_data/gmm_results_seven_pcs.csv")

    print(results_gmm_five_pcs)
