import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Sample DataFrame
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [2, 4, 6, 8, 10],
    'C': [5, np.nan, np.nan, 8, 10]
}

df = pd.DataFrame(data)

# Function to impute missing values using Linear Regression
def impute_missing_values(df, target_column):
    # Separate the rows with and without missing target values
    not_null_df = df[df[target_column].notna()]
    null_df = df[df[target_column].isna()]
    
    if null_df.empty:
        return df
    
    # Features (excluding the target column)
    X_train = not_null_df.drop(columns=[target_column])
    y_train = not_null_df[target_column]
    
    # Features of rows with missing target values
    X_pred = null_df.drop(columns=[target_column])
    
    # Impute missing values in the input features
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_pred_imputed = imputer.transform(X_pred)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train_imputed, y_train)
    
    # Predict the missing values
    y_pred = model.predict(X_pred_imputed)
    
    # Fill the missing values with the predictions
    df.loc[df[target_column].isna(), target_column] = y_pred
    
    return df

# Impute missing values for each column with missing data
for column in df.columns:
    df = impute_missing_values(df, column)

print("DataFrame after imputation:")
print(df)
