import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import networkx as nx
import sklearn
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.models import LinearMediation, Prediction

def CausalDiscovery(dataset, target, tau = 5, type = 0):
    r"""
    This function prepares a time series dataset for prediction using the PCMCI framework (Predictive Causal Modeling with Conditional Independence).
    It employs the Tigramite library for prediction, using either ParCorr or CMIknn for conditional independence testing.

    Parameters
    ----------
    - dataset (pandas.DataFrame): The dataset containing time series data. Columns represent variables, and rows represent time steps.
    - target (str): The name of the target variable (column) in the dataset for which predictions are made.
    - tau (int): The maximum time lag considered when selecting predictors (default = 5).
    - type (int): An integer that determines the type of conditional independence test to use (default = 0).
      - 0: Uses `ParCorr` (Partial Correlation Test).
      - Else: Uses `CMIknn` (Conditional Mutual Information test with nearest neighbors).

    Returns
    -------
    - tuple: A tuple containing two elements:
      - true_data (numpy.ndarray): The true target values from the test set.
      - predicted (numpy.ndarray): The predicted target values generated by the fitted model.
    """
        
    # Convert the dataset (likely a pandas DataFrame or a similar data structure) into a NumPy array.
    # This extracts only the raw values from the dataset, excluding any metadata such as column labels.
    data_array = dataset.values

    # Create a dataframe compatible with the PCMCI library using the data array.
    # The DataFrame is created by the pp ('tigramite.data_processing') and takes two parameters:
    # - data: The data to populate the DataFrame, which is the NumPy array created earlier (data_array).
    # - var_names: The variable names (column names) are passed as a list of the original dataset's columns.
    dataframe = pp.DataFrame(data=data_array, var_names=list(dataset.columns))

    if type == 0:
        cond_ind_test = ParCorr()
    else:
        # Create an instance of the CMIknn class for conditional independence testing.
        # - significance='fixed_thres': Specifies that a fixed threshold method will be used for significance testing.
        # - model_selection_folds=3: Indicates that 3-fold cross-validation will be used for model selection.
        # - knn=0.05: Number of nearest-neighbors which determines the size of hyper-cubes around each (high-dimensional) sample point.
        #            If smaller than 1, this is computed as a fraction of T. For knn larger or equal to 1, this is the absolute number.
        cond_ind_test = CMIknn(knn = 0.05, significance='fixed_thres', model_selection_folds=3)

    # Get the total number of time steps from the shape of the data array.
    T = data_array.shape[0]

    # Create an instance of the Prediction class for making predictions using the specified model and settings.
    # Parameters:
    # - dataframe: The input data to be used for making predictions.
    # - cond_ind_test: The conditional independence test method to use
    # - prediction_model: The machine learning model to use for prediction, here it is set to LinearRegression from scikit-learn.
    # - data_transform: The data transformation method to standardize the data before prediction, here it is set to StandardScaler from scikit-learn.
    # - train_indices: Indices for training data, which includes the first 70% of the time steps.
    # - test_indices: Indices for test data, which includes the last 30% of the time steps.
    # - verbosity: Controls the level of output information during execution. A value of 1 provides intermediate-level details.
    pred = Prediction(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        prediction_model=sklearn.linear_model.LinearRegression(),
        data_transform=sklearn.preprocessing.StandardScaler(),
        train_indices=range(int(0.7 * T)),
        test_indices=range(int(0.3 * T), T),
        verbosity=1
    )

    # Set the target variable index and maximum time lag for prediction.
    self_target = dataset.columns.get_loc(target)
    tau_max = tau

    # Get the predictors for the specified target variable.
    # - selected_targets: List of target variables to predict. Here, we have a single target variable with index 9.
    # - steps_ahead: The number of steps ahead to predict. Here, it's set to 1.
    # - tau_max: The maximum time lag to consider for predictors. Here, it's set to 5.
    # - pc_alpha: Significance level for the conditional independence test. Here, it's set to 0.05.
    predictors = pred.get_predictors(
        selected_targets=[self_target],
        steps_ahead=1,
        tau_max=tau_max,
        pc_alpha=0.05
    )

    # Fit the prediction model using the specified predictors and target.
    # - target_predictors: The predictors for the target variable, obtained from the previous step.
    # - selected_targets: The target variables for which predictions are to be made. Here, it is a single target with index `target`.
    # - tau_max: The maximum time lag to consider for prediction. This should match the value used when getting predictors.
    pred.fit(
        target_predictors=predictors,  # Predictors for the target variable.
        selected_targets=[self_target],      # The target variable to predict.
        tau_max=tau_max                 # Maximum time lag considered for the prediction.
    )

    # Use the fitted prediction model to make predictions for the specified target variable.
    # The `predict` method generates predictions for the target variable using the trained model.
    predicted = pred.predict(self_target)

    # Retrieve the true test data for the target variable.
    # The `get_test_array` method returns the true values for the specified target variable from the test set.
    true_data = pred.get_test_array(j=self_target)[0]

    return true_data, predicted
