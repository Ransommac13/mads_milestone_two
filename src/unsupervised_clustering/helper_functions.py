"""

feature_set.py

This module includes helper functions for exploring unsupervised models.
"""

import textwrap
import re
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.config import RANDOM_SEED


CLUSTERING_FEATURES = ['competitors_within_1_mi', 'median_household_income',
       'total_population', 'age_18_29_proportion', 'poverty_rate',
       'bachelors_degree_proportion_25plus', 'owner_occupied_proportion',
       'family_households_proportion', 'no_vehicle_households_proportion',
       'D1C', 'D3A', 'NatWalkInd', 'commercial_places_within_1_mi', 'rac_c000',
       'wac_ce01_prop', 'consumer_facing_jobs_proportion',
       'industrial_and_log_jobs_proportion', 'aadt_count']

def inspect_skewness(imputed_df: pd.DataFrame):

    skewness = imputed_df.skew().sort_values(key=abs, ascending=False)

    return skewness.to_frame()

def eval_explained_variance(pca, feature_names):

    explained_variance = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(pca.n_components_)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)
    })

    pc_loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    return explained_variance, pc_loadings

def scree_plot(pca):

    fig, ax1 = plt.subplots(figsize=(10, 5))

    components = np.arange(1, pca.n_components_ + 1)

    # Individual explained variance
    ax1.bar(
        components,
        pca.explained_variance_ratio_,
        alpha=0.6,
        label="Individual variance"
    )

    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_xticks(components)

    # Cumulative explained variance
    ax2 = ax1.twinx()

    ax2.plot(
        components,
        np.cumsum(pca.explained_variance_ratio_),
        marker="o",
        label="Cumulative variance"
    )

    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_ylim(0, 1.05)

    plt.title("PCA Explained Variance")
    plt.tight_layout()
    plt.show()

def pc_loading_heatmap(pc_loadings: pd.DataFrame, n_components=9):

    def format_label(label):
        label = str(label)

        # 1. Remove leading log1p indicators
        label = re.sub(r"^(?:log1p[_\s]*|log1p__)", "", label, flags=re.I)
        # 2. Replace proportion with prop
        label = label.replace("proportion", "prop")
        # 3. Replace underscores with spaces and wrap long labels
        label = label.replace("_", " ")
        if len(label) > 20:
            label = textwrap.fill(label, width=20)
        return label

    # Apply formatting without modifying original DataFrame
    plot_data = pc_loadings.iloc[:, :n_components].copy()
    plot_data.index = [format_label(col) for col in plot_data.index]
    plt.figure(figsize=(12, 9))

    sns.heatmap(
        plot_data,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f"
    )

    plt.title("PCA Component Coefficients")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

#def pc_loading_heatmap(pc_loadings: pd.DataFrame, n_components=9):

    #plt.figure(figsize=(12, 9))

    #sns.heatmap(
    #    pc_loadings.iloc[:, :n_components],
    #    cmap="coolwarm",
    #    center=0,
    #    annot=True,
    #    fmt=".2f"
    #)

    #plt.title("PCA Component Coefficients")
    #plt.tight_layout()
    #plt.show()

def pc_v_pc_scatterplot(X_pca, train_df: pd.DataFrame, pc_x=1, pc_y=2, color_by=None, kind='scatter', agg_func=np.median):

    x_col = f"PC{pc_x}"
    y_col = f"PC{pc_y}"

    # Select only the columns we need
    cols = ["agtno"]
    if color_by is not None:
        cols.append(color_by)

    # Match PCA scores with retailer attributes
    plot_df = X_pca.merge(
        train_df[cols],
        on="agtno",
        how="left",
        validate="one_to_one"
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter Plot
    if kind == "scatter":
        if color_by is None:
            ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.4, s=10)

        elif pd.api.types.is_numeric_dtype(plot_df[color_by]):
            sc = ax.scatter(plot_df[x_col], plot_df[y_col], c=plot_df[color_by], cmap="viridis", alpha=0.5, s=10)
            fig.colorbar(sc, ax=ax, label=color_by)

        else:
            sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=color_by, alpha=0.5, s=12, ax=ax)
            ax.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Hexbin
    elif kind == "hex":
        # No color variable -> hex color = point count
        if color_by is None:
            hb = ax.hexbin(plot_df[x_col], plot_df[y_col], gridsize=40, cmap="viridis", mincnt=1)
            fig.colorbar(hb, ax=ax, label="Retailer count")

        # Numeric color variable -> hex color = aggregated value
        elif pd.api.types.is_numeric_dtype(plot_df[color_by]):
            hb = ax.hexbin(plot_df[x_col], plot_df[y_col], C=plot_df[color_by], reduce_C_function=agg_func, gridsize=40, cmap="viridis", mincnt=1)
            label = f"{agg_func.__name__} {color_by}"
            fig.colorbar(hb, ax=ax, label=label)

        # Categorical color variable -> not ideal for hexbin
        else:
            raise ValueError("Hexbin mode works best with no color_by or a numeric color_by."
                "For categorical variables, use kind='scatter' or create separate plots by category.")

    else:
        raise ValueError("kind must be either 'scatter' or 'hex'")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    if kind == "hex" and color_by is not None and pd.api.types.is_numeric_dtype(plot_df[color_by]):
        ax.set_title(f"{x_col} vs {y_col} (hexbin colored by {agg_func.__name__} {color_by})")
    elif kind == "hex":
        ax.set_title(f"{x_col} vs {y_col} (hexbin)")
    else:
        ax.set_title(f"{x_col} vs {y_col}")

    plt.tight_layout()
    plt.show()

def plot_gmm_results(results_df, metric='bic'):

    """
    Plot GMM evaluation metrics by number of components, with separate lines for each covariance structure.

    Parameters:
        results_df: DataFrame of GMM evaluation results
        metric: 'bic' or 'val_log_likelihood'
    """

    if metric not in ["bic", "val_log_likelihood"]:
        raise ValueError("metric must be 'bic' or 'val_log_likelihood'")

    plt.figure(figsize=(8, 5))

    sns.lineplot(data=results_df, x="n_components", y=metric, hue="covariance", marker="o")

    plt.xlabel("Number of Components")
    plt.ylabel("BIC" if metric == "bic" else "Validation Log-Likelihood")
    plt.title(f"GMM Model Selection: {metric}")
    plt.xticks(sorted(results_df["n_components"].unique()))
    plt.legend(title="Covariance Structure")
    plt.tight_layout()
    plt.show()

def gmm_cluster_profile_heatmap(assignments: pd.DataFrame, train_df: pd.DataFrame):

    assert len(assignments) == len(train_df)
        
    profile_df = train_df.merge(assignments, on='agtno', how='inner', validate='one_to_one')

    overall_median = profile_df[CLUSTERING_FEATURES].median()
    overall_iqr = profile_df[CLUSTERING_FEATURES].quantile(0.75) - profile_df[CLUSTERING_FEATURES].quantile(0.25)

    cluster_medians = profile_df.groupby("cluster")[CLUSTERING_FEATURES].median()
    standardized = (cluster_medians - overall_median) / overall_iqr.replace(0, np.nan)

    plt.figure(figsize=(10,7))
    sns.heatmap(standardized.T, cmap='coolwarm', center=0, vmin=-2, vmax=2, annot=True, fmt=".1f", linewidths=0.5)

    plt.title("Standardized GMM Component Profiles")
    plt.xlabel("Cluster")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return standardized

def gmm_cluster_categorical_dist(assignments: pd.DataFrame, train_df: pd.DataFrame, cat_var='region',
                                 color_center=0.5, vmin=0.0, vmax=1.0):

    assert len(assignments) == len(train_df)
            
    profile_df = train_df.merge(assignments, on='agtno', how='inner', validate='one_to_one')

    cat_dist = pd.crosstab(profile_df['cluster'], profile_df[cat_var], normalize='index').round(3)

    plt.figure(figsize=(10,7))
    sns.heatmap(cat_dist.T, cmap='coolwarm', center=color_center, vmin=vmin, vmax=vmax, annot=True, fmt=".3f", linewidths=0.5)

    plt.title("Cluster Assignment by Region")
    plt.xlabel("Cluster")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.show()

    return cat_dist

def gmm_cluster_numerical_dist(assignments: pd.DataFrame, train_df: pd.DataFrame, clusters: list, features: list,
                               ncols=2, figsize_per_plot=(6,4)):

    assert len(assignments) == len(train_df)
    profile_df = train_df.merge(assignments, on='agtno', how='inner', validate='one_to_one')

    plot_df = profile_df[profile_df['cluster'].isin(clusters)].copy()
    nrows = math.ceil(len(features)/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_plot[0]*ncols, figsize_per_plot[1]*nrows), squeeze=False)
    axes=axes.flatten()

    for i, feature in enumerate(features):
        sns.violinplot(data=plot_df,x='cluster', y=feature, order=clusters, inner="quart", cut=0, ax=axes[i])

        axes[i].set_title(feature)
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel("")

    # Remove unused subplots
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Feature Distributions: Clusters {clusters}", fontsize=14, y=1.01)

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_gmm_profiles(gmm, X_train: pd.DataFrame, X_val: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols=CLUSTERING_FEATURES, figsize=(12, 8)):
    # Add GMM assignments to original, preprocessed feature data
    train_assignments = pd.DataFrame({
        "agtno": X_train["agtno"].to_numpy(),
        "cluster": gmm.predict(X_train.drop(columns="agtno"))
    })

    val_assignments = pd.DataFrame({
        "agtno": X_val["agtno"].to_numpy(),
        "cluster": gmm.predict(X_val.drop(columns="agtno"))
    })

    assert len(train_assignments) == len(train_df)
    assert len(val_assignments) == len(val_df)
    
    train_profile = train_df.merge(train_assignments, on="agtno", validate="one_to_one")
    val_profile = val_df.merge(val_assignments, on="agtno", validate="one_to_one")

    # Reference statistics come exclusively from training data
    train_median = train_profile[feature_cols].median()
    train_iqr = (train_profile[feature_cols].quantile(0.75) - train_profile[feature_cols].quantile(0.25)).replace(0, np.nan)

    train_cluster_medians = (train_profile.groupby("cluster")[feature_cols].median())

    val_cluster_medians = (val_profile.groupby("cluster")[feature_cols].median())

    clusters = range(gmm.n_components)

    train_cluster_medians = train_cluster_medians.reindex(clusters)
    val_cluster_medians = val_cluster_medians.reindex(clusters)

    # Standardize both using the same training reference
    train_standardized = (train_cluster_medians - train_median) / train_iqr
    val_standardized = (val_cluster_medians - train_median) / train_iqr

    # Difference: validation minus training
    profile_difference = val_standardized - train_standardized

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for ax, data, title in zip(axes,[train_standardized.T, val_standardized.T, profile_difference.T],
                               ["Training Profiles", "Validation Profiles","Validation_Training"]):
        
        is_difference = title == "Validation_Training"

        sns.heatmap(data, cmap="coolwarm", center=0, vmin=-1 if is_difference else -2, vmax=1 if is_difference else 2, ax=ax, cbar=True)

        ax.set_title(title)
        ax.set_xlabel("Cluster")

    axes[0].set_ylabel("Feature")

    plt.tight_layout()
    plt.show()

    return train_standardized, val_standardized, profile_difference

def compare_gmm_confidence(gmm, X_train: pd.DataFrame, X_val: pd.DataFrame):

    train_probs = gmm.predict_proba(X_train)
    val_probs = gmm.predict_proba(X_val)

    confidence_df = pd.concat([
        pd.DataFrame({'dataset': 'Train', 'cluster': train_probs.argmax(axis=1), 'max_prob': train_probs.max(axis=1)}),
        pd.DataFrame({'dataset': 'Val', 'cluster': val_probs.argmax(axis=1), 'max_prob': val_probs.max(axis=1)})
    ])

    overall_df = confidence_df.groupby('dataset')['max_prob'].agg(
        count='count',
        mean='mean',
        median='median',
        std='std'
    )

    low_confidence = (confidence_df.assign(
        below_60=confidence_df["max_prob"] < 0.60,
        below_80=confidence_df["max_prob"] < 0.80)
        .groupby("dataset")[["below_60", "below_80"]].mean()
    )

    overall_df = overall_df.join(low_confidence)

    # Confidence within each component
    by_cluster = (confidence_df.groupby(["dataset", "cluster"])["max_prob"].agg(["count", "mean", "median"]))

    # Violin Plot
    plt.figure(figsize=(9,5))

    sns.violinplot(data=confidence_df, x="dataset", y="max_prob",
        order=["Train", "Val"], inner="quart", cut=0)

    plt.title("GMM Assignment Confidence: Train vs. Validation")
    plt.ylabel("Maximum Membership Probability")
    plt.xlabel("")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

    return overall_df, by_cluster, confidence_df

def plot_kmeans_silhouette(representations, k_range=range(2, 13), sample_size=3000):
    """
    Compare K-means silhouette scores across feature representations.

    Parameters
    representations : Dictionary mapping representation names to feature matrices. Example: {"5 PCs": train_pca5, "7 PCs": train_pca7, "10 PCs": train_pca10}

    k_range : Numbers of clusters to evaluate.

    sample_size : Number of observations sampled for silhouette calculation. Set to None for exact scores, with considerable computational cost.

    Returns DataFrame of silhouette scores for each representation and k.
    """

    results = []

    for name, X in representations.items():

        X = np.asarray(X)
        for k in k_range:
            model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            labels = model.fit_predict(X)
            n_sample = (min(sample_size, len(X)) if sample_size is not None else None)

            score = silhouette_score(X, labels, sample_size=n_sample, random_state=RANDOM_SEED)

            results.append({
                "representation": name,
                "k": k,
                "silhouette_score": score
            })

    results_df = pd.DataFrame(results)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))

    sns.lineplot(
        data=results_df,
        x="k",
        y="silhouette_score",
        hue="representation",
        marker="o",
        linewidth=2,
        markersize=6,
        ax=ax
    )

    ax.set_title("K-means Silhouette Scores by Feature Representation", fontsize=13, pad=15)

    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Mean Silhouette Score", fontsize=11)

    ax.set_xticks(list(k_range))

    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(title="Feature Representation", frameon=False)

    plt.tight_layout()
    plt.show()

    return results_df

def plot_kmeans_pca_scatter(X_pca, n_clusters=2, n_components=5, point_size=9, alpha=0.45, show_lowess=True, lowess_frac=0.3):
    """
    Fit K-means using the selected number of PCs and plot cluster membership in the PC1-PC2 plane.

    Parameters
    X_pca : PCA-transformed observations.
    n_clusters : Number of K-means clusters.
    n_components : Number of PCs used to FIT K-means. Only PC1 and PC2 are displayed.
    point_size : Scatterplot marker size.
    alpha : Marker transparency.

    Returns
    model : Fitted K-means model.
    labels : Cluster assignments.
    """

    X_pca_df = X_pca.copy()
    X_pca = np.asarray(X_pca)
    X_model = X_pca[:, :n_components]
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)

    labels = model.fit_predict(X_model)

    # Consistent categorical colors
    palette = sns.color_palette("tab10", n_colors=n_clusters)
    colors = np.array(palette)[labels]

    fig, ax = plt.subplots(figsize=(8,5.5))

    # Scatter plot
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=point_size, alpha=alpha, linewidths=0, rasterized=True)

    if show_lowess:
            smoothed = lowess(endog=X_pca_df["PC2"], exog=X_pca_df["PC1"], frac=lowess_frac, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", linewidth=2.2, alpha=0.85, linestyle="--", zorder=5)

    # Project fitted centroids onto scatter plot.
    centers = model.cluster_centers_[:, :2]

    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=180, c="black", edgecolors="white", linewidths=1, zorder=5, label="Centroids")

    # Legend handles
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=palette[i], markeredgecolor="none", markersize=7, label=f"Cluster {i}") for i in range(n_clusters)]
    handles.append(Line2D([0], [0], marker="X", linestyle="", color="black", markersize=9, label="Centroids"))
    if show_lowess:
            handles.append(Line2D([0], [0], color="black", linewidth=2.2, linestyle="--", label="LOWESS trend"))

    ax.legend(handles=handles, title="K-means Membership", frameon=False, loc="best")
    ax.set_ylim(top=10)
    ax.set_title(f"K-means Clusters in PCA Space (k={n_clusters})", fontsize=13, pad=15)
    ax.set_xlabel("Principal Component 1", fontsize=11)
    ax.set_ylabel("Principal Component 2", fontsize=11)

    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()

    return model, labels
