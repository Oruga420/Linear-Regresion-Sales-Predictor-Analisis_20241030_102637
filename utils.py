import pandas as pd
import numpy as np

def validate_data(df):
    """
    Validate the input dataframe has the required columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to validate
    
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    required_columns = ['Marketing_Spend', 'Price', 'Sales']
    return all(col in df.columns for col in required_columns)

def preprocess_data(df):
    """
    Preprocess the data for model training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to preprocess
    
    Returns:
    --------
    tuple
        (X, y) where X is the feature matrix and y is the target variable
    """
    # Select features and target
    X = df[['Marketing_Spend', 'Price']]
    y = df['Sales']
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    return X, y
