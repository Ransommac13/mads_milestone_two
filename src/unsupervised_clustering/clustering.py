"""

clustering.py

This module makes Reeses's peanut butter clusters (gross!).
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from itertools import combinations
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import RANDOM_SEED, TRAIN_DATA_PARQUET_PATH, VALIDATION_DATA_PARQUET_PATH
from src.config import TRAIN_PCA_PARQUET_PATH, VAL_PCA_PARQUET_PATH, TEST_PCA_PARQUET_PATH
from src.config import GMM_PARQUET_DIR
from src.unsupervised_clustering.preprocess import CLUSTERING_FEATURES
from src.unsupervised_clustering.preprocess import wac_feature_eng, select_features, build_preprocessing_pipeline, main_pipepline
from src.unsupervised_clustering.helper_functions import gmm_cluster_profile_heatmap, gmm_cluster_categorical_dist, gmm_cluster_numerical_dist
from src.unsupervised_clustering.helper_functions import compare_gmm_profiles, compare_gmm_confidence

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

def fit_single_gmm(X_train: pd.DataFrame, n_components: int, agtno, covariance_type = 'full', n_init=30, random_state=RANDOM_SEED):
    "This function fits a single GMM and returns the model, assignments, and membership probabilities."

    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, n_init=n_init, reg_covar=1e-5, random_state=random_state)
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

def compare_gmm_assignments(assignments_a: pd.DataFrame, assignments_b: pd.DataFrame):

    comparison = assignments_a[["agtno", "cluster"]].merge(assignments_b[["agtno", "cluster"]],
        on="agtno", validate="one_to_one", suffixes=("_a", "_b"))

    counts = pd.crosstab(comparison["cluster_b"], comparison["cluster_a"])

    row_pct = pd.crosstab(comparison["cluster_b"], comparison["cluster_a"], normalize="index").round(3)

    return counts, row_pct

def evaluate_gmm_initialization_stability(X_train: pd.DataFrame, n_components=9, seeds=range(20),
                                          covariance_type='full', n_init=5, reg_covar=1e-5):
    models = {}
    assignments = {}
    fit_results = []

    for seed in seeds:
        model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, n_init=n_init, reg_covar=reg_covar, random_state=seed)
        model.fit(X_train)

        models[seed] = model
        assignments[seed] = model.predict(X_train)

        fit_results.append({
            'seed': seed,
            'bic': model.bic(X_train),
            'train_log_likelihood': model.score(X_train),
            'converged': model.converged_
        })

    ari_results = []
    for seed_a, seed_b in combinations(seeds, 2):
        ari = adjusted_rand_score(assignments[seed_a], assignments[seed_b])
        ari_results.append({'seed_a': seed_a, 'seed_b': seed_b, 'ari': ari})

    fit_results_df = pd.DataFrame(fit_results)
    ari_results_df = pd.DataFrame(ari_results)

    return fit_results_df, ari_results_df, models

def compare_to_best_gmm(fit_results, fitted_models, X_train):

    best_seed = fit_results.loc[
        fit_results["bic"].idxmin(), "seed"
    ]

    reference_labels = fitted_models[best_seed].predict(X_train)

    results = fit_results.copy()

    results["ari_to_best"] = results["seed"].apply(
        lambda seed: adjusted_rand_score(
            reference_labels,
            fitted_models[seed].predict(X_train)
        )
    )

    results["delta_bic"] = results["bic"] - results["bic"].min()

    return results.sort_values("bic")

def compare_gmm_prevalence(gmm, X_train: pd.DataFrame, X_val: pd.DataFrame):

    train_labels = gmm.predict(X_train)
    val_labels = gmm.predict(X_val)

    components = range(gmm.n_components)

    train_pct = (pd.Series(train_labels).value_counts(normalize=True).reindex(components, fill_value=0))

    val_pct = (pd.Series(val_labels).value_counts(normalize=True).reindex(components, fill_value=0))

    comparison = pd.DataFrame({"train_pct": train_pct, "val_pct": val_pct})

    comparison["difference_pp"] = (comparison["val_pct"] - comparison["train_pct"]) * 100

    return comparison

def export_gmm_features(gmm, X_pca_train: pd.DataFrame, X_pca_val: pd.DataFrame, X_pca_test: pd.DataFrame, output_dir, num_pcs=5):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {'train': X_pca_train, 'val': X_pca_val, 'test': X_pca_test}

    pc_cols = [f"PC{i}" for i in range(1, num_pcs+1)]

    outputs = {}
    for split_name, df in datasets.items():

        assert df["agtno"].notna().all()
        assert df["agtno"].is_unique
        assert df[pc_cols].notna().all().all()

        X = df[pc_cols]
        cluster_probs = gmm.predict_proba(X)

        prob_cols = [f"gmm_prob_{k}" for k in range(gmm.n_components)]
        result = pd.DataFrame(cluster_probs, columns=prob_cols)
        
        result.insert(0, 'agtno', df['agtno'].to_numpy())
        result['gmm_cluster'] = cluster_probs.argmax(axis=1)
        result['gmm_max_prob'] = cluster_probs.max(axis=1)

        assert len(result) == len(df)
        assert np.allclose(result[prob_cols].sum(axis=1), 1)
        assert result["agtno"].is_unique

        filepath = output_dir / f"X_gmm_{split_name}.parquet"
        result.to_parquet(filepath, index=False)

        outputs[split_name] = result
        print(f"Saved X_gmm_{split_name} of shape {result.shape} to {str(output_dir)}")

    return outputs


if __name__ == '__main__':

    X_train_raw = pd.read_parquet(TRAIN_DATA_PARQUET_PATH)
    X_val_raw = pd.read_parquet(VALIDATION_DATA_PARQUET_PATH)
    
    X_train_processed = pd.read_parquet("src/unsupervised_clustering/processed_data/X_train_processed.parquet")
    X_val_processed = pd.read_parquet("src/unsupervised_clustering/processed_data/X_val_processed.parquet")

    X_pca_train_w_agtno = pd.read_parquet(TRAIN_PCA_PARQUET_PATH)
    X_pca_val_w_agtno = pd.read_parquet(VAL_PCA_PARQUET_PATH)
    X_pca_test_w_agtno = pd.read_parquet(TEST_PCA_PARQUET_PATH)

    X_pca_train = X_pca_train_w_agtno.drop(columns='agtno')
    X_pca_val = X_pca_val_w_agtno.drop(columns='agtno')
    pca_train_agtno = X_pca_train_w_agtno['agtno']
    pca_val_agtno = X_pca_val_w_agtno['agtno']

    train_pca5 = X_pca_train[[f"PC{i}" for i in range(1,6)]]
    val_pca5 = X_pca_val[[f"PC{i}" for i in range(1,6)]]

    #results_gmm_original = evaluate_gmm(X_train_processed, X_val_processed).sort_values("bic")
    #results_gmm_five_pcs = evaluate_gmm(train_pca5, val_pca5).sort_values("bic")

    # Initial model shortlist: (train_pca5, 5), (train_pca5, 9), (train_pca5, 11), (train_pca7, 9), (train_pca7, 12), (X_train_processed, 11) 
    #fit_results, ari_results, fitted_models = evaluate_gmm_initialization_stability(train_pca5, n_components=5, seeds=range(25), covariance_type='full', n_init=30)

    # After evaluating for stability: (train_pca5, 5, n_init=30, random_state=4)
    gmm_primary, assignments_primary = fit_single_gmm(train_pca5, 5, pca_train_agtno, covariance_type='full', n_init=30, random_state=4)
    
    # Evaluate component profiles using preimputed data.
    preprocessor = joblib.load("src/unsupervised_clustering/fitted_objects/preprocess.joblib")
    train_profile_full = preprocessor.named_steps['wac_engineering'].transform(X_train_raw) # Train data with engineered wac features.
    train_profile_trimmed = train_profile_full[CLUSTERING_FEATURES + ['agtno']]
    val_profile_full = preprocessor.named_steps['wac_engineering'].transform(X_val_raw)
    val_profile_trimmed = val_profile_full[CLUSTERING_FEATURES + ['agtno']]

    train_pca5_profile = X_pca_train_w_agtno[[f"PC{i}" for i in range(1,6)] + ['agtno']]
    val_pca5_profile = X_pca_val_w_agtno[[f"PC{i}" for i in range(1,6)] + ['agtno']]

    #train_profiles, val_profiles, profile_diffs = compare_gmm_profiles(gmm_primary, train_pca5_profile, val_pca5_profile,
    #                                                                   train_profile_trimmed, val_profile_trimmed)

    #overall_df, by_cluster_df, confidence_df = compare_gmm_confidence(gmm_primary, train_pca5, val_pca5)

    #final_gmm_feature_dfs = export_gmm_features(gmm_primary, X_pca_train_w_agtno, X_pca_val_w_agtno, X_pca_test_w_agtno,
    #                                            output_dir=GMM_PARQUET_DIR, num_pcs=5)
