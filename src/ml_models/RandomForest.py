import numpy as np
from matplotlib import pyplot as plt
import sklearn
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def RandomForest(dataframe, features, target, n_splits=0):
    r"""
    Preprocesses the data, trains a Random Forest Regressor, and returns predictions on the test set. 
    If n_splits > 1, performs K-Fold Cross Validation.

    Parameters
    ----------
    - dataframe (pandas.DataFrame): A DataFrame containing the input features and the target variable.
    - features (list): A list of column names representing the features used as input for the model.
    - target (str): The name of the column in the DataFrame that represents the target variable.
    - n_splits (int, optional): Number of folds for K-Fold CV. If 0 or None, no K-Fold is applied.

    Returns
    -------
    - tuple: A tuple containing two elements:
        - y_test (numpy.ndarray): The true target values from the test set.
        - pred (numpy.ndarray): The predicted target values from the test set.

    The function performs the following steps:
    1. Copies the input 'dataframe' to a new variable 'self_dataframe' for internal use.
    2. Initializes a MinMaxScaler to scale the features between 0 and 1.
    3. Iterates through the features (excluding the target variable), scaling each feature using the MinMaxScaler.
    4. Converts the preprocessed DataFrame into a NumPy array:
       - 'X' contains the scaled feature data (excluding the target).
       - 'Y' contains the target variable data.
    5. If n_splits is 0 or not specified, splits the data into training and test sets (70% training, 30% testing). Otherwise, applies K-Fold Cross Validation 
        with the specified number of folds, splitting the data into training and test sets for each fold.
    6. Initializes a Random Forest Regressor with 100 decision trees and a fixed random seed.
    7. Trains the model using the training data (X_train and y_train).
    8. Uses the trained model to predict the target values for the test set (X_test).
    9. Returns the actual test target values (y_test) and the predicted values (pred) for evaluation.
    """
    
    df = dataframe.copy()
    sc = MinMaxScaler(feature_range=(0, 1))

    for var in features:
        if var != target:
            df[var] = sc.fit_transform(df[var].values.reshape(-1, 1))

    X = df.drop(columns=target).to_numpy()
    Y = df[target].to_numpy()

    sc_Y = MinMaxScaler(feature_range=(0,1))
    Y_scaled = sc_Y.fit_transform(Y.reshape(-1, 1)).flatten()

    seed = 7
    np.random.seed(seed)

    if n_splits and n_splits > 1:
        # K-Fold Cross Validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        all_y_true = []
        all_y_pred = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            regressor = RandomForestRegressor(n_estimators=100, random_state=seed)
            regressor.fit(X_train, y_train)

            pred = regressor.predict(X_test)
            all_y_true.append(y_test)
            all_y_pred.append(pred)

        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        return y_true, y_pred

    else:
        # Base behavior: single train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

        regressor = RandomForestRegressor(n_estimators=100, random_state=seed)
        regressor.fit(X_train, y_train)
        pred = regressor.predict(X_test)
        return y_test, pred