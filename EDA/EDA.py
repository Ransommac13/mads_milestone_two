#Read-in training dataset

import pandas as pd

df2 = pd.read_parquet('mads_milestone_two/data/final/master_data.parquet')
#print(df2)

#df2.to_csv('/Users/brisabol/Documents/MADS/SIADS 696/dataexplore_EDA.csv', index=True)

#print(df2.columns)

#subset to feature columns
#df3 = df2[columns]

summary = df2.describe()

# Add a row at the top showing the missing value count
summary.loc['missing_values'] = df2.isna().sum()

#summary.to_csv('/Users/brisabol/Documents/MADS/SIADS 696/dataexplore_EDA2.csv', index=True)

#hide row agtno = 113165
df3 = df2.drop(index=4984)
#df3 = df3[df3["avg_daily_sales"] <= 10000]
#print(df2.iloc[4984])

import matplotlib.pyplot as plt

#column = "competitors_within_1_mi"

#plt.hist(df3[column].dropna(), bins=20, edgecolor="black")
#plt.xlabel(column)
#plt.ylabel("Frequency")
#plt.title(f"Distribution of {column}")
#plt.show()
'''
numeric_columns = df2.select_dtypes(include="number").columns

df2[numeric_columns].hist(
    bins=10,
    figsize=(15, 10),
    edgecolor="black"
)

plt.suptitle("Distributions of Numeric Columns")
plt.tight_layout()
plt.show()
'''
#need to produce scatterplot of predictors versus outcome var

#numeric_cols = df3.select_dtypes(include="number").columns
'''
x_max = 10_000

fig, axes = plt.subplots(
    nrows=len(numeric_columns),
    ncols=1,
    figsize=(10, 3 * len(numeric_columns)),
    squeeze=False,
)

for ax, col in zip(axes.flat, numeric_columns):
    ax.scatter(df3["avg_daily_sales"], df3[col], alpha=0.7)
    ax.set_xlim(0, x_max)
    ax.set_title(col)
    ax.set_xlabel("avg_daily_sales")
    ax.set_ylabel(col)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
'''
'''
plots_per_page = 5
x_max = 10_000
y_max = 10_000

for start in range(0, len(numeric_columns), plots_per_page):
    columns_to_plot = numeric_columns[start:start + plots_per_page]

    fig, axes = plt.subplots(
        nrows=len(columns_to_plot),
        ncols=1,
        figsize=(10, 3 * len(columns_to_plot)),
        squeeze=False,
    )

    for ax, col in zip(axes.flat, columns_to_plot):
        ax.scatter(df3["avg_daily_sales"], df3[col], alpha=0.7)

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)

        ax.set_title(f"{col} vs. avg_daily_sales")
        ax.set_xlabel("avg_daily_sales")
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
  

plots_per_page = 5

for start in range(0, len(numeric_columns), plots_per_page):
    columns_to_plot = numeric_columns[start:start + plots_per_page]

    fig, axes = plt.subplots(
        nrows=len(columns_to_plot),
        ncols=1,
        figsize=(10, 3 * len(columns_to_plot)),
        squeeze=False,
    )

    for ax, col in zip(axes.flat, columns_to_plot):
        x = df3["avg_daily_sales"]
        y = df3[col]

        ax.scatter(x, y, alpha=0.7)

        # Set each axis to the corresponding column's min and max
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        ax.set_title(f"{col} vs. avg_daily_sales")
        ax.set_xlabel("avg_daily_sales")
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plots_per_page = 5

for start in range(0, len(numeric_columns), plots_per_page):
    columns_to_plot = numeric_columns[start:start + plots_per_page]

    fig, axes = plt.subplots(
        nrows=len(columns_to_plot),
        ncols=1,
        figsize=(10, 3 * len(columns_to_plot)),
        squeeze=False,
    )

    for ax, col in zip(axes.flat, columns_to_plot):
        plot_data = (
            df3[["avg_daily_sales", col]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        x = plot_data["avg_daily_sales"]
        y = plot_data[col]

        if len(plot_data) > 1 and x.nunique() > 1 and y.nunique() > 1:
            sns.kdeplot(
                data=plot_data,
                x="avg_daily_sales",
                y=col,
                fill=True,
                cmap="viridis",
                levels=20,
                thresh=0.05,
                ax=ax,
            )

            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
        else:
            ax.text(
                0.5,
                0.5,
                "Not enough variation for density plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title(f"Density of {col} vs. avg_daily_sales")
        ax.set_xlabel("avg_daily_sales")
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    '''

#looking at columns to determin what type of missing value imputation
'''
'agtno'                     (NA)
'avg_daily_sales'           (no missing values)
'chain'                     (string)
'bustype'                   (string)
'longitude'                 (NA)
'latitude',                 (NA)
'bg_geoid'                  (NA)                
'tract_geoid'               (NA)
'competitors_within_1_mi'   (no missing values)
'aadt_count'                (over half missing)
'aadt_functional_class'     (over half missing?)
'aadt_lanes'                (over half missing)
'dist_to_read_m'            (over half missing)
'median_household_income'   (1495/12846 missing; mean impute)
'total_population'          (95/14246 missing; mean impute)
'age_18_29_proportion'      (167/14174; mean impute)
'age_30_64_proportion'      (167/14174; mean impute)
'poverty_rate'              (189/14152; median impute)
'bachelors_degree_proportion_25plus'    (168/14173; mean impute)
'post_grad_degree_proportion_25plus'    (168/14173; median impute)
'labor_force_participation_rate'        (167/14174; mean impute)
'unemployment_rate'                     (178/14163; median impute)
'owner_occupied_proportion'             (189/14152; median impute)
'family_households_proportion'          (189/14152; mean impute)
'no_vehicle_households_proportion'      (189/14152; median impute)
'D1B'                                   (2654/11687; median impute approx 22% missing)
'D1C'                                   (2654/11687; median impute)
'D3A'                                   (2654/11687; mean impute)
'NatWalkInd'                            (2654/11687; mean impute)
'commercial_places_within_1_mi'         (44/14297; median impute)
'community_places_within_1_mi'          (44/14297; median impute)
'rac_c000'                              (6/14335; median impute)
'rac_ce01_prop'                         (6/14335; mean impute)
'rac_ce02_prop'                         (6/14335; mean impute)
'wac_cns01'                             (10/14331; median impute)
'wac_cns02'                             (10/14331; median impute)
'wac_cns03'                             (10/14331; median impute)
'wac_cns04'                             (10/14331; median impute)
'wac_cns05'                             (10/14331; median impute)
'wac_cns06'                             (10/14331; median impute)
'wac_cns07'                             (10/14331; median impute)
'wac_cns08'                             (10/14331; median impute)
'wac_cns09'                             (10/14331; median impute)
'wac_cns10'                             (10/14331; median impute)
'wac_cns11'                             (10/14331; median impute)
'wac_cns12'                             (10/14331; median impute)
'wac_cns13'                             (10/14331; median impute)
'wac_cns14'                             (10/14331; median impute)
'wac_cns15'                             (10/14331; median impute)
'wac_cns16'                             (10/14331; median impute)
'wac_cns17'                             (10/14331; median impute)
'wac_cns18'                             (10/14331; median impute)
'wac_cns19'                             (10/14331; median impute)
'wac_cns20'                             (10/14331; median impute)
'wac_c000'                              (10/14331; median impute)
'wac_ce01_prop'                         (10/14331; mean impute)
'wac_ce02_prop'                         (10/14331; mean impute)
'''