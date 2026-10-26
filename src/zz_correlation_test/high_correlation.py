from json import load

from matplotlib.pyplot import flag
from narwhals import exclude
from numpy import dtype
import pandas as pd
import argparse
import numpy as np

from pathlib import Path
from src.config import TRAIN_DATA_PARQUET_PATH, PROJECT_ROOT, TRAINING_MASTER_PARQUET_PATH, FLAG_CORRELATION_CSV_PATH

OUTPUT_PATH = PROJECT_ROOT / FLAG_CORRELATION_CSV_PATH

def load_data(path):
    """
    load the training data in modeling folder, only numeric features

    """
    
    path = path
    df = pd.read_parquet(path)

    print(df)

    # leave longitude and latitude out of the correlation check 
    exclude_col = ["longitude", "latitude"]

    numeric = df.select_dtypes(include="number")
    numeric = numeric.drop(columns=exclude_col, errors="ignore")

    return numeric

def build_correlation_matrix(numeric_df, method):

    #Using pearson method 
    method = "pearson"
    correlation = numeric_df.corr(method)

    return correlation

def flag_high_correlation_matrix(correlation_matrix, threshold):
    """
    returns every unique feature pair at or above the threshold 
    using numpy triu *triangle , upper)
    """
    # threshold = 0.8
    # Only use the upper half, to avoid being counted twice
    triangle_upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1))

    # turn matrix into rows with the two variables and its correlation
    flagged = triangle_upper.stack().rename("correlation").reset_index()
    flagged.columns = ["variable_1", "variable_2", "correlation"]
    
    flagged["abs_correlation"] = flagged["correlation"].abs()
    
    # filter the correlation with selected threshold
    flagged = flagged[flagged["abs_correlation"] >= threshold ]

    #round to three decimal places
    flagged["abs_correlation"] = round(flagged["abs_correlation"], 3)
    flagged["correlation"] = round(flagged['correlation'], 3)
    
    # sort by strongest correlation first
    sorted_flag = flagged.sort_values("abs_correlation", ascending=False).reset_index(drop=True)

    return sorted_flag

def find_correlated_clusters(flagged_pairs):

    clusters = []

    # Go through each row of correlated variable pairs
    for _, row in flagged_pairs.iterrows():

        # Make the two variables in this row to a set
        set = {row["variable_1"], row["variable_2"]}

        #F Find clusters that shares at least one of the variable with this pair
        overlap = [c for c in clusters if c & set]

        # if pair belongs to an existing cluster
        # update each overlapping cluster into the pair, then remove the old cluster from the list
        for c in overlap: 
            
            set.update(c)
            clusters.remove(c)

        clusters.append(set)
    
    return clusters

def main(threshold=0.8):

    # Load data 
    numeric_df = load_data(TRAIN_DATA_PARQUET_PATH)

    # Calculate correlation maxtrix
    corr_matrix = build_correlation_matrix(numeric_df, method="pearson")

    # Flag correlaion
    flagged = flag_high_correlation_matrix(corr_matrix, threshold)

    #check numeric variables
    print(f"{numeric_df.shape[1]} numeric features")

    print(f"len{flagged} variable pairs at or above |r| = {threshold}: \n")
    print(flagged.to_string(index=False))

    clusters = find_correlated_clusters(flagged)
    if clusters: 
        print(f" {len(clusters)} correlated feature cluster(s):")
        for i, cluster in enumerate(clusters, 1):
            print(f" {i}. {sorted(cluster)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok = True)
    flagged.to_csv(OUTPUT_PATH, index=False)

    return flagged

if __name__ == "__main__":
    main()