import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def train_linear_regression(X, y, test_size=0.2, random_state=42):
    """
    Train a linear regression model on the provided data.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    tuple
        (model, X_train, X_test, y_train, y_test, y_pred)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred

def get_model_metrics(model, X_test, y_test):
    """
    Calculate model performance metrics.
    
    Parameters:
    -----------
    model : LinearRegression
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test target values
    
    Returns:
    --------
    dict
        Dictionary containing model metrics
    """
    y_pred = model.predict(X_test)
    return {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred)
    }
