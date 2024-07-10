import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Sample DataFrame with missing values
data = {'A': [1, 2, np.nan, 4, 5], 'B': [np.nan, 2, 3, 4, 5]}
df = pd.DataFrame(data)

# KNN Imputation
imputer = KNNImputer(n_neighbors=2)
df_knn_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("KNN Imputed DataFrame:")
print(df_knn_imputed)
