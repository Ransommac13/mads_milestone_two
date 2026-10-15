"""

feature_set.py

This module defines the feature set by which we want to define a lottery retailer's potential "archetype."
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#from src.correlation_test.high_correlation import build_correlation_matrix, flag_high_correlation_matrix

def evaluate_correlation(clustering_df: pd.DataFrame, method="spearman", threshold = 0.75):

    correlation_df = build_correlation_matrix(clustering_df, method=method)
    high_correlation_df = flag_high_correlation_matrix(correlation_df, threshold=threshold)

    return high_correlation_df


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

def pc_loading_heatmap(pc_loadings: pd.DataFrame):

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        pc_loadings.iloc[:, :10],
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f"
    )

    plt.title("PCA Component Coefficients")
    plt.tight_layout()
    plt.show()

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
