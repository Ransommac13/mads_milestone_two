"""

sales_clean.py
Input: sales_data.parquet
Output: average_daily_sales.parquet

This module takes raw sales figures per retailer-day, groups by retailer (agtno), and aggregates by average daily sales (mean).
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import SALES_PARQUET_PATH

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

    # Compute number of calendar days each retailer existed in 2025.
    annual_sales_df['num_cal_days'] = (annual_sales_df['last_date'] - annual_sales_df['first_date']).dt.days + 1
    # Compute the target variable average daily sales.
    annual_sales_df['avg_daily_sales'] = (annual_sales_df['total_sales'] / annual_sales_df['num_cal_days'])
    
    return annual_sales_df

if __name__ == '__main__':

    # Uncomment the following line to read in the full dataset.
    sales_df = pd.read_parquet(SALES_PARQUET_PATH)

    # Uncomment the following four lines to read in a sample of the dataset for troubleshooting code. 
    #lottery_sales_file_path = Path(SALES_PARQUET_PATH)
    #sales_pq = pq.ParquetFile(lottery_sales_file_path)
    #sales_first_n_rows = next(sales_pq.iter_batches(batch_size=50000))
    #sales_df = pa.Table.from_batches([sales_first_n_rows]).to_pandas()

    annual_sales_df = groupby_agg_sales_data(sales_df)
    print(annual_sales_df.sample(10))
    print(len(annual_sales_df))
