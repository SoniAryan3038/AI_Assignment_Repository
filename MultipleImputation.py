import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Sample DataFrame with missing values
data = {'A': [1, 2, np.nan, 4, 5], 'B': [np.nan, 2, 3, 4, 5]}
df = pd.DataFrame(data)

# Multiple Imputation using Iterative Imputer
imputer = IterativeImputer(max_iter=10, random_state=0)
df_multiple_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("Multiple Imputed DataFrame:")
print(df_multiple_imputed)
