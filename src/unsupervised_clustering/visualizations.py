"""

visualizations.py

Pretty pictures!
This module includes functions to create the core visualizations used for unsupervised model evaluation and understanding.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
import joblib
from adjustText import adjust_text

from src.config import TRAIN_PCA_PARQUET_PATH, GMM_PARQUET_DIR, TRAIN_DATA_PARQUET_PATH
from src.unsupervised_clustering.a_preprocess import wac_feature_eng, select_features, build_preprocessing_pipeline, main_pipepline

def pca_biplot(X_pca_train: pd.DataFrame, pca_model, feature_names, n_features=8, pc_x=1, pc_y=2):

    """
    Plot retailer PCA scores with arrows showing feature-PC correlations.

    Parameters
    X_pca : DataFrame of PCA scores containing columns such as PC1, PC2, etc.
    pca : fitted PCA model object.
    feature_names : list of names of input variables in the exact order used by PCA.
    n_features : Number of highest-contributing features to display.
    pc_x, pc_y : Principal component numbers to use as the x and y axes.
    """

    x_idx = pc_x - 1
    y_idx = pc_y - 1
    x_col = f"PC{pc_x}"
    y_col = f"PC{pc_y}"

    # Feature-PC correlations for standardized PCA inputs
    correlations = (pca.components_.T * np.sqrt(pca.explained_variance_))
    # Select contributions across the displayed PCs
    arrow_magnitude = np.sqrt(correlations[:, x_idx] ** 2 + correlations[:, y_idx] ** 2)
    selected_idx = np.argsort(arrow_magnitude)[-n_features:]

    # Retrieve retailer coordinates
    x = X_pca_train[x_col].to_numpy()
    y = X_pca_train[y_col].to_numpy()

    # Scale arrows into the score coordinate system
    x_range = np.percentile(x, 99) - np.percentile(x, 1)
    y_range = np.percentile(y, 99) - np.percentile(y, 1)

    scale = (0.35 * min(x_range, y_range) * 1)

    fig, ax = plt.subplots(figsize=(9,6))

    # Retailer scores
    ax.scatter(x, y, s=7, alpha=0.12, color="gray", edgecolors="none", rasterized=True)

    # Reference lines
    ax.axhline(0, color="lightgray", linewidth=0.8)
    ax.axvline(0, color="lightgray", linewidth=0.8)

    texts = []
    for idx in selected_idx:
        dx = correlations[idx, x_idx] * scale
        dy = correlations[idx, y_idx] * scale

        ax.annotate("", xy=(dx, dy), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color='darkred', lw=1.5, mutation_scale=12))

        # Collect labels rather than leaving them at fixed positions
        texts.append(ax.text(dx * 1.08, dy * 1.08, feature_names[idx], fontsize=9, color='darkred', ha="left" if dx >= 0 else "right", va="center"))

    ax.set_xlabel(f"PC{pc_x} ({pca.explained_variance_ratio_[x_idx]:.1%} variance)")
    ax.set_ylabel(f"PC{pc_y} ({pca.explained_variance_ratio_[y_idx]:.1%} variance)")
    ax.set_title("PCA Biplot: Retailer Environmental Characteristics")
    ax.set_ylim((-4,6))
    ax.set_xlim((-4,6))

    adjust_text(texts, ax=ax, target_x=[correlations[idx, x_idx] * scale for idx in selected_idx],
                target_y=[correlations[idx, y_idx] * scale for idx in selected_idx],
                prevent_crossings=True, only_move={"text": "y", "static": "y", "explode": "y", "pull": "y"},
                arrowprops=dict(arrowstyle="-", color="black", lw=0.7, alpha=0.6))

    sns.despine()
    fig.tight_layout()

    return fig, ax

def gmm_pca_plot(X_pca_train: pd.DataFrame, X_gmm_train: pd.DataFrame, explained_variance, n_clusters=5, show_lowess=True, lowess_frac=0.3):

    pca_plot = X_pca_train[['agtno', 'PC1', 'PC2']]
    gmm_plot = X_gmm_train[['agtno', 'gmm_cluster', 'gmm_max_prob']]

    plot_df = pca_plot.merge(gmm_plot, on='agtno', validate='one_to_one')
    assert len(plot_df) == len(X_pca_train)
    plot_df = plot_df.sort_values('gmm_max_prob')

    palette = sns.color_palette('colorblind', n_clusters)
    # Assign color based on cluster.
    cluster_colors = np.array(palette)[plot_df["gmm_cluster"].to_numpy(dtype=int)]
    # Assign alpha based on membership confidence
    alpha = np.clip(0.25 + 0.65 * plot_df["gmm_max_prob"].to_numpy(), 0.25, 0.90)
    # Combine RGB and alpha into RGBA array
    colors = np.column_stack([cluster_colors, alpha])

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.scatter(plot_df["PC1"], plot_df["PC2"], s=10, c=colors, edgecolors="none", rasterized=True)

    if show_lowess:
        smoothed = lowess(endog=plot_df["PC2"], exog=plot_df["PC1"], frac=lowess_frac, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", linewidth=2.2, alpha=0.85, linestyle="--", zorder=5)

    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} variance explained)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} variance explained)")

    # Manual legend because the scatter uses per-point colors
    handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=palette[k], markeredgecolor="none", markersize=7, label=f"Component {k}") for k in range(n_clusters)]
    if show_lowess:
        handles.append(Line2D([0], [0], color="black", linewidth=2.2, linestyle="--", label="LOWESS trend"))

    ax.legend(handles=handles, title="Dominant GMM component", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(top=10)

    sns.despine()
    fig.tight_layout()

    return fig, ax

def region_pca_plot(X_pca_train: pd.DataFrame, X_train: pd.DataFrame, explained_variance, alpha=0.4):
    """
    Plot PC1 vs. PC2, colored by geographic region.

    Parameters
    X_pca_train: PCA scores containing agtno, PC1, and PC2.
    X_train : Original training data containing agtno and region.
    explained_variance : Explained variance ratios for principal components.
    alpha : Point transparency.
    """

    pca_plot = X_pca_train[['agtno', 'PC1', 'PC2']]
    region_plot = X_train[['agtno', 'region']]

    plot_df = pca_plot.merge(region_plot, on='agtno', validate='one_to_one')

    assert len(plot_df) == len(X_pca_train)

    # Consistent region ordering
    regions = sorted(plot_df['region'].dropna().unique())
    # One color per region
    palette = sns.color_palette('husl', len(regions))
    region_colors = dict(zip(regions, palette))
    # Map each observation to its region's color
    colors = plot_df['region'].map(region_colors)

    fig, ax = plt.subplots(figsize=(9,5.5))

    ax.scatter(plot_df['PC1'], plot_df['PC2'],
        s=10, c=colors.tolist(), alpha=alpha, edgecolors='none', rasterized=True)

    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} variance explained)")

    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} variance explained)")

    # Manual legend
    handles = [Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=region_colors[region], markeredgecolor='none', markersize=7, label=region) for region in regions]
    
    ax.legend(handles=handles, title="Region", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(top=10)

    sns.despine()
    fig.tight_layout()

    return fig, ax

if __name__ == '__main__':

    X_train = pd.read_parquet(TRAIN_DATA_PARQUET_PATH)
    X_pca_train = pd.read_parquet(TRAIN_PCA_PARQUET_PATH)
    gmm_parquet_dir = Path(GMM_PARQUET_DIR)
    X_gmm_train_3_8 = pd.read_parquet(gmm_parquet_dir / "X_gmm_train_3PCs.parquet")
    X_gmm_train_5_5 = pd.read_parquet(gmm_parquet_dir / "X_gmm_train.parquet")
    pca = joblib.load("src/unsupervised_clustering/fitted_objects/pca.joblib")
    preprocessor = joblib.load("src/unsupervised_clustering/fitted_objects/preprocess.joblib")

    original_feature_names = preprocessor.named_steps['numeric_preprocessing'].get_feature_names_out()
    original_feature_names = [s.removeprefix("other_cols__") for s in original_feature_names]
    original_feature_names = ["log_" + s[20:] if s.startswith("log_transform_") else s for s in original_feature_names]

    fig, ax = gmm_pca_plot(X_pca_train, X_gmm_train_3_8, pca.explained_variance_ratio_, n_clusters=8, show_lowess=False, lowess_frac=0.5)
    plt.show()