#Read-in training dataset

import pandas as pd

df2 = pd.read_parquet('mads_milestone_two/data/final/master_data.parquet')
#print(df2)

#df2.to_csv('/Users/brisabol/Documents/MADS/SIADS 696/dataexplore_EDA.csv', index=False)

#questions: where is the outcome variable?

#print(df2.columns)

columns = ['chain', 'bustype', 'competitors_within_1_mi', 'aadt_count', 'aadt_functional_class', 'aadt_lanes', 
           'dist_to_read_m', 'median_household_income', 'total_population', 'age_18_29_proportion', 
           'age_30_64_proportion', 'poverty_rate', 'bachelors_degree_proportion_25plus', 'post_grad_degree_proportion_25plus', 
           'labor_force_participation_rate', 'unemployment_rate', 'owner_occupied_proportion', 'family_households_proportion', 
           'no_vehicle_households_proportion', 'D1B', 'D1C', 'D3A', 'NatWalkInd', 'commercial_places_within_1_mi', 
           'community_places_within_1_mi', 'rac_c000', 'rac_ce01', 'rac_ce02', 'wac_c000', 'wac_cns07', 'wac_cns18', 
           'wac_ce01', 'wac_ce02']

#subset to feature columns
df3 = df2[columns]

summary = df3.describe()

# Add a row at the top showing the missing value count
summary.loc['missing_values'] = df3.isna().sum()

summary.to_csv('/Users/brisabol/Documents/MADS/SIADS 696/dataexplore_EDA2.csv', index=True)

#Initial thoughts: figure out how to handle missing values, maybe columns aadt_count and dist_to_read_m should be removed
#since more than half are missing...

df_sales = pd.read_parquet('mads_milestone_two/data/final/sales_data.parquet')
#print(df_sales)

#print(df_sales.columns)
#print(len(df_sales))

#length train data is 16740, length of sales data is 14351
print(df3)


import matplotlib.pyplot as plt

#column = "competitors_within_1_mi"

#plt.hist(df3[column].dropna(), bins=20, edgecolor="black")
#plt.xlabel(column)
#plt.ylabel("Frequency")
#plt.title(f"Distribution of {column}")
#plt.show()

numeric_columns = df3.select_dtypes(include="number").columns

df3[numeric_columns].hist(
    bins=10,
    figsize=(15, 10),
    edgecolor="black"
)

plt.suptitle("Distributions of Numeric Columns")
plt.tight_layout()
plt.show()
'''
numeric_cols = df3.select_dtypes(include="number").columns

fig, axes = plt.subplots(
    nrows=len(numeric_cols),
    ncols=1,
    figsize=(10, 3 * len(numeric_cols)),
    squeeze=False,
)

for ax, col in zip(axes.flat, numeric_cols):
    ax.scatter(df3.index, df3[col], alpha=0.7)
    ax.set_title(col)
    ax.set_xlabel("Index")
    ax.set_ylabel(col)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
'''