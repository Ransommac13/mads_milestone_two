"""

clustering.py

This module makes Reeses's peanut butter clusters (gross!).
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from src.config import RANDOM_SEED
from src.unsupervised_clustering.pca import pc_v_pc_scatterplot

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

def evaluate_gmm(X_train, X_val, k_range=(1,13)):
    
    results = []
    for k in k_range:
        for covariance in ["full", "diag", "tied"]:
            model = GaussianMixture(n_components=k, covariance_type=covariance, n_init=5, reg_covar=1e-5, random_state=RANDOM_SEED)
            model.fit(X_train)

            results.append({
                "k": k,
                "covariance": covariance,
                "bic": model.bic(X_train),
                "val_log_likelihood": model.score(X_val)
            })

    return pd.DataFrame(results)

if __name__ == '__main__':

    original_scaled_data = pd.read_parquet("src/unsupervised_clustering/eda_data/scaled_data.parquet")
    X_pca = pd.read_parquet("src/unsupervised_clustering/eda_data/X_pca.parquet")
    X_pca = X_pca.drop(columns='agtno')

    first_five_pcs = X_pca[[f"PC{i}" for i in range(1,6)]]
    first_seven_pcs = X_pca[[f"PC{i}" for i in range(1,8)]]
    first_ten_pcs = X_pca[[f"PC{i}" for i in range(1,11)]]

    #results_kmeans_original = evaluate_kmeans(original_scaled_data)
    #results_kmeans_five_pcs = evaluate_kmeans(first_five_pcs)
    #results_kmeans_seven_pcs = evaluate_kmeans(first_seven_pcs)

    