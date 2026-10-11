#Read-in training dataset

import pandas as pd

df2 = pd.read_parquet('mads_milestone_two/data/final/master_data.parquet')
#print(df2)

#df2.to_csv('/Users/brisabol/Documents/MADS/SIADS 696/dataexplore_EDA.csv', index=False)

#print(df2.columns)

columns = ['agtno', 'avg_daily_sales', 'chain', 'bustype', 'competitors_within_1_mi', 'aadt_count',
       'aadt_functional_class', 'aadt_lanes', 'dist_to_read_m',
       'median_household_income', 'total_population', 'age_18_29_proportion',
       'age_30_64_proportion', 'poverty_rate',
       'bachelors_degree_proportion_25plus',
       'post_grad_degree_proportion_25plus', 'labor_force_participation_rate',
       'unemployment_rate', 'owner_occupied_proportion',
       'family_households_proportion', 'no_vehicle_households_proportion',
       'D1B', 'D1C', 'D3A', 'NatWalkInd', 'commercial_places_within_1_mi',
       'community_places_within_1_mi', 'rac_c000', 'rac_ce01_prop',
       'rac_ce02_prop', 'wac_cns01', 'wac_cns02', 'wac_cns03', 'wac_cns04',
       'wac_cns05', 'wac_cns06', 'wac_cns07', 'wac_cns08', 'wac_cns09',
       'wac_cns10', 'wac_cns11', 'wac_cns12', 'wac_cns13', 'wac_cns14',
       'wac_cns15', 'wac_cns16', 'wac_cns17', 'wac_cns18', 'wac_cns19',
       'wac_cns20', 'wac_c000', 'wac_ce01_prop', 'wac_ce02_prop']

#subset to feature columns
df3 = df2[columns]

summary = df3.describe()

# Add a row at the top showing the missing value count
summary.loc['missing_values'] = df3.isna().sum()

summary.to_csv('/Users/brisabol/Documents/MADS/SIADS 696/dataexplore_EDA2.csv', index=True)

#hide row agtno = 113165
#df3 = df3.drop(index=4986)
#df3 = df3[df3["avg_daily_sales"] != 593051.3767123290]

import matplotlib.pyplot as plt

#column = "competitors_within_1_mi"

#plt.hist(df3[column].dropna(), bins=20, edgecolor="black")
#plt.xlabel(column)
#plt.ylabel("Frequency")
#plt.title(f"Distribution of {column}")
#plt.show()

numeric_columns = df3.select_dtypes(include="number").columns
'''
df3[numeric_columns].hist(
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

plots_per_page = 5
x_max = 10_000
y_max = 5_000

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