"""

sales_clean.py
Input: sales_data.parquet
Output: average_daily_sales.parquet

This module takes raw sales figures per retailer-day, groups by retailer (agtno), and aggregates by average daily sales (mean).
"""

from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import SALES_PARQUET_PATH, FINAL_SALES_PARQUET_PATH

def groupby_agg_sales_data(sales_df: pd.DataFrame):

    # First inspect for missingness. 
    assert not sales_df.isna().any().any(), "DataFrame contains missing values"

    # Convert sales figures from string to float. 
    sales_columns = ['numbers_day', 'numbers_eve',
       'numbers_iw', 'win4_day', 'win4_eve', 'win4_iw', 't5_day', 't5_eve',
       'take5_iw', 'pick10', 'lotto', 'mega', 'megaplier', 'jtj', 'powerball',
       'powerplay', 'doubleplay', 'c4l', 'quick_draw', 'qd_extra',
       'money_dots', 'promo', 'draw_paid', 'ig_settles', 'ig_paid',
       'game0_mm_iw', 'game1', 'game2', 'game3', 'game4']
    
    sales_df[sales_columns] = sales_df[sales_columns].astype(float)

    # Convert bus_day to datetime. 
    sales_df['bus_day'] = pd.to_datetime(sales_df['bus_day'])

    # Sum sales for each retailer-day.
    sales_df['total_sales'] = sales_df[sales_columns].sum(axis=1)

    # Group by agtno. Aggregate by total sales and identify first date and last date.
    annual_sales_df = sales_df.groupby('agtno').agg(
        total_sales=('total_sales', 'sum'),
        first_date=('bus_day', 'min'),
        last_date=('bus_day', 'max')
    ).reset_index()

    # Identify first and last dates with positive sales to distinguish inactive periods.
    positive_sales_df = sales_df.loc[sales_df['total_sales'] > 0].groupby('agtno').agg(
        first_pos_date=('bus_day', 'min'),
        last_pos_date=('bus_day', 'max'),
        pos_sales_days=('bus_day', 'nunique')
    ).reset_index()

    # Merge both dataframes.
    annual_sales_df = annual_sales_df.merge(positive_sales_df, on='agtno', how='left')

    # Compute number of calendar days and active days each retailer existed in 2025.
    annual_sales_df['num_cal_days'] = (annual_sales_df['last_date'] - annual_sales_df['first_date']).dt.days + 1
    annual_sales_df['num_active_days'] = (annual_sales_df['last_pos_date'] - annual_sales_df['first_pos_date']).dt.days + 1
    # Compute the number of potentially inactive days before and after first and last sales.
    annual_sales_df['days_before_first_sale'] = (annual_sales_df['first_pos_date'] - annual_sales_df['first_date']).dt.days
    annual_sales_df['days_after_last_sale'] = (annual_sales_df['last_date'] - annual_sales_df['last_pos_date']).dt.days
    # Compute the target variable average daily sales, both by calendar days and active days.
    annual_sales_df['avg_cal_daily_sales'] = (annual_sales_df['total_sales'] / annual_sales_df['num_cal_days'])
    annual_sales_df['avg_active_daily_sales'] = (annual_sales_df['total_sales'] / annual_sales_df['num_active_days'])
    
    return annual_sales_df

def process_annual_sales_data(annual_df: pd.DataFrame, min_cal_days=30, inactivity_threshold=21):

    annual_df = annual_df.copy()

    # Filter to retailers with one than one month of calendar days on record.
    annual_df = annual_df.loc[annual_df['num_cal_days'] >= min_cal_days].copy()
    # Filter to retailers that sold at least one lottery ticket during the year.
    annual_df = annual_df.loc[annual_df['total_sales'] > 0].copy()

    # Distinguish retailers that appear to not be actively selling lottery tickets at some point during the year.
    annual_df["use_active_period"] = (
        (annual_df["days_before_first_sale"] > inactivity_threshold) |
        (annual_df["days_after_last_sale"] > inactivity_threshold)
    )

    # Select average daily sales value based on whether they were active for the entire period on record.
    annual_df['avg_daily_sales'] = np.where(
        annual_df['use_active_period'],
        annual_df['avg_active_daily_sales'],
        annual_df['avg_cal_daily_sales']
    )

    return annual_df

if __name__ == '__main__':

    # Uncomment the following line to read in the full dataset.
    sales_df = pd.read_parquet(SALES_PARQUET_PATH)

    # Uncomment the following four lines to read in a sample of the dataset for troubleshooting code. 
    #lottery_sales_file_path = Path(SALES_PARQUET_PATH)
    #sales_pq = pq.ParquetFile(lottery_sales_file_path)
    #sales_first_n_rows = next(sales_pq.iter_batches(batch_size=500000))
    #sales_df = pa.Table.from_batches([sales_first_n_rows]).to_pandas()

    annual_sales_df = groupby_agg_sales_data(sales_df)

    processed_annual_sales_df = process_annual_sales_data(annual_sales_df)
    print(processed_annual_sales_df.shape)
    print(processed_annual_sales_df.sample(5))

    final_sales_path = Path(FINAL_SALES_PARQUET_PATH)
    final_sales_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_annual_sales_df.to_parquet(str(final_sales_path), index=False, engine="pyarrow")
