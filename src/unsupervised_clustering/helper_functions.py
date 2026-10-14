"""

feature_set.py

This module defines the feature set by which we want to define a lottery retailer's potential "archetype."
"""

import pandas as pd

from src.correlation_test.high_correlation import build_correlation_matrix, flag_high_correlation_matrix

def evaluate_correlation(clustering_df: pd.DataFrame, method="spearman", threshold = 0.75):

    correlation_df = build_correlation_matrix(clustering_df, method=method)
    high_correlation_df = flag_high_correlation_matrix(correlation_df, threshold=threshold)

    return high_correlation_df


def inspect_skewness(imputed_df: pd.DataFrame):

    skewness = imputed_df.skew().sort_values(key=abs, ascending=False)

    return skewness.to_frame()
