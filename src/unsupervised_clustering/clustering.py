"""

clustering.py

This module makes Reeses's peanut butter clusters (gross!).
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import joblib

from src.config import RANDOM_SEED, TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH
from src.unsupervised_clustering.preprocess import CLUSTERING_FEATURES
from src.unsupervised_clustering.preprocess import wac_feature_eng, select_features, build_preprocessing_pipeline, main_pipepline

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

def fit_single_gmm(X_train: pd.DataFrame, n_components: int, agtno, covariance_type = 'full'):
    "This function fits a single GMM and returns the model, assignments, and membership probabilities."

    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, n_init=5, reg_covar=1e-5, random_state=RANDOM_SEED)
    model.fit(X_train)

    probabilities = model.predict_proba(X_train)

    assignments = pd.DataFrame({
        "agtno": agtno.to_numpy(),
        "cluster": model.predict(X_train),
        "max_prob": probabilities.max(axis=1)
    }, index=X_train.index)

    for k in range(n_components):
        assignments[f"prob_cluster_{k}"] = probabilities[:, k]

    return model, assignments

def evaluate_gmm_components(model, assignments: pd.DataFrame):

    summary = (assignments.groupby("cluster").agg(
            retailer_count=("cluster", "size"),
            mean_max_prob=("max_prob", "mean"),
            median_max_prob=("max_prob", "median")
        ))

    summary["retailer_pct"] = (100 * summary["retailer_count"] / len(assignments))

    summary["mixture_weight"] = model.weights_
    summary["mixture_weight"] = 100 * summary["mixture_weight"]

    return summary

def component_profile_table(assignments: pd.DataFrame, train_df: pd.DataFrame):

    assert len(assignments) == len(train_df)
    
    merged_df = train_df.merge(assignments, on='agtno', how='inner', validate='one_to_one')

    candidate_cols_to_drop = ['avg_daily_sales', 'longitude', 'latitude', 'bg_geoid', 'tract_geoid']
    cols_to_drop = [col for col in candidate_cols_to_drop if col in merged_df.columns]
    merged_df = merged_df.drop(columns=cols_to_drop)

    numeric_cols = merged_df.select_dtypes(include='number').columns.drop('cluster', errors='ignore')
    numeric_cols = [col for col in numeric_cols if col != 'agtno' and not col.startswith('prob_cluster_') and col != "max_probability"]
    categorical_cols = merged_df.select_dtypes(include=['object', 'category', 'string', 'bool']).columns
    categorical_cols = [col for col in categorical_cols if col not in ['agtno', 'cluster']]

    numeric_profile = merged_df.groupby('cluster')[numeric_cols].median()
    categorical_profile = merged_df.groupby('cluster')[categorical_cols].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA
    )

    profile_table = pd.concat([numeric_profile, categorical_profile], axis=1)
    profile_table = profile_table.T

    return profile_table


if __name__ == '__main__':

    X_train_raw = pd.read_parquet(TRAIN_DATA_PARQUET_PATH)
    X_val_raw = pd.read_parquet(VALIDATION_DATA_PARQUET_PATH)
    
    X_train_processed = pd.read_parquet("src/unsupervised_clustering/eda_data/X_train_processed.parquet")
    X_val_processed = pd.read_parquet("src/unsupervised_clustering/eda_data/X_val_processed.parquet")

    X_pca_train_w_agtno = pd.read_parquet("src/unsupervised_clustering/eda_data/X_pca_train.parquet")
    X_pca_val_w_agtno = pd.read_parquet("src/unsupervised_clustering/eda_data/X_pca_val.parquet")
    X_pca_train = X_pca_train_w_agtno.drop(columns='agtno')
    X_pca_val = X_pca_val_w_agtno.drop(columns='agtno')
    pca_train_agtno = X_pca_train_w_agtno['agtno']
    pca_val_agtno = X_pca_val_w_agtno['agtno']

    train_pca5 = X_pca_train[[f"PC{i}" for i in range(1,6)]]
    val_pca5 = X_pca_val[[f"PC{i}" for i in range(1,6)]]
    train_pca7 = X_pca_train[[f"PC{i}" for i in range(1,8)]]
    val_pca7 = X_pca_val[[f"PC{i}" for i in range(1,8)]]
    
    #results_kmeans_original = evaluate_kmeans(X_train_processed)
    #results_kmeans_five_pcs = evaluate_kmeans(train_pca5)
    #results_kmeans_seven_pcs = evaluate_kmeans(train_pca7)

    #results_gmm_original = evaluate_gmm(X_train_processed, X_val_processed).sort_values("bic")
    #results_gmm_five_pcs = evaluate_gmm(train_pca5, val_pca5).sort_values("bic")
    #results_gmm_seven_pcs = evaluate_gmm(train_pca7, val_pca7).sort_values("bic")

    #results_gmm_original.to_csv("src/unsupervised_clustering/eda_data/gmm_results_original.csv")
    #results_gmm_five_pcs.to_csv("src/unsupervised_clustering/eda_data/gmm_results_five_pcs.csv")
    #results_gmm_seven_pcs.to_csv("src/unsupervised_clustering/eda_data/gmm_results_seven_pcs.csv")

    # Initial model shortlist: (train_pca5, 5), (train_pca5, 9), (train_pca5, 11), (train_pca7, 9), (train_pca7, 12), (X_train_processed, 11) 
    gmm_5pc_9clust, assignments_5pc_9clust = fit_single_gmm(train_pca5, 9, pca_train_agtno)
    #gmm_original_11clust, assignments_original_11clust = fit_single_gmm(X_train_processed, 11, X_train_raw['agtno'])
    #summary_5pc_9clust = evaluate_gmm_components(gmm_5pc_9clust, assignments_5pc_9clust)

    # Evaluate component profiles using preimputed data.
    preprocessor = joblib.load("src/unsupervised_clustering/preprocess.joblib")
    profile_df_full = preprocessor.named_steps['wac_engineering'].transform(X_train_raw) # Train data with engineered wac features.
    profile_df_trimmed = profile_df_full[CLUSTERING_FEATURES + ['agtno']]

    # Trimmed model shortlist: (train_pca5, 5), (train_pca5, 9), (train_pca7, 9), (X_train_processed, 11) 
    profile_5pc_9clust = component_profile_table(assignments_5pc_9clust, profile_df_full)
    #profile_original_11clust = component_profile_table(assignments_original_11clust, profile_df_trimmed)

    print(profile_5pc_9clust)
