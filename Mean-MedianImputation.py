import pandas as pd
import numpy as np

# Sample DataFrame with missing values
data = {'A': [1, 2, np.nan, 4, 5], 'B': [np.nan, 2, 3, 4, 5]}
df = pd.DataFrame(data)

# Mean Imputation
df_mean_imputed = df.copy()
df_mean_imputed['A'].fillna(df_mean_imputed['A'].mean(), inplace=True)
df_mean_imputed['B'].fillna(df_mean_imputed['B'].mean(), inplace=True)

print("Mean Imputed DataFrame:")
print(df_mean_imputed)

# Median Imputation
df_median_imputed = df.copy()
df_median_imputed['A'].fillna(df_median_imputed['A'].median(), inplace=True)
df_median_imputed['B'].fillna(df_median_imputed['B'].median(), inplace=True)

print("Median Imputed DataFrame:")
print(df_median_imputed)
