import numpy as np
from matplotlib import pyplot as plt
import sklearn
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import metrics


def create_model_large(size, X_train):
    r"""
    Creates and compiles a large artificial neural network model for regression tasks.

    Parameters
    ----------
    - size (int): The number of neurons in the input and the last hidden layer. 
                  This determines the complexity and capacity of the model.
    - X_train (numpy.ndarray): The training feature data used to determine the input dimension 
                               of the model.

    Returns
    ----------
    - model (keras.Sequential): A compiled Sequential model ready for training.

    The function performs the following operations:
    1. Initializes a Sequential model, allowing layers to be stacked sequentially.
    2. Adds an input layer with a specified number of neurons (defined by 'size') and 
       ReLU (Rectified Linear Unit) as the activation function. The input dimension is set 
       to the number of features in the training data (X_train).
    3. Adds a hidden layer with double the number of neurons compared to the input layer 
       (2 * size) and uses ReLU activation.
    4. Adds another hidden layer with the same number of neurons as the input layer (size) 
       and ReLU activation.
    5. Adds an output layer with a single neuron (suitable for regression tasks) and no 
       activation function.
    6. Compiles the model using the Adam optimizer, with mean squared error as the loss 
       function and mean absolute error (MAE) as a performance metric.

    The compiled model is then returned for training.
    """
    # Create a Sequential model, which allows layers to be stacked one after another.
    model = Sequential()
    
    # Add the input layer with 35 neurons and ReLU activation function.
    # The 'input_dim' is set to the number of features in the training data (X_train).
    model.add(Dense(size, input_dim=X_train.shape[1], activation='relu'))
    
    # Add a hidden layer with 70 neurons and ReLU activation function.
    model.add(Dense(size*2, activation='relu'))
    
    # Add another hidden layer with 35 neurons and ReLU activation function.
    model.add(Dense(size, activation='relu'))
    
    # Add the output layer with a single neuron (for regression tasks) and no activation function.
    model.add(Dense(1))
    
    # Compile the model with the Adam optimizer, mean squared error as the loss function, and mean absolute error (MAE) as a performance metric.
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metrics.mae])
    
    # Return the compiled model.
    return model

def ANN(dataframe, features, target):
    r"""
    Constructs and trains an artificial neural network (ANN) to predict a specific target 
    based on a provided set of features.

    Parameters
    ----------
    - dataframe (pandas.DataFrame): A DataFrame containing the data, including both features and the target.
    - features (list): A list of column names representing the features used as inputs to the neural network.
    - target (str): The name of the column in the DataFrame that represents the target variable to be predicted.

    Returns
    ----------
    - tuple: A tuple containing two elements:
        - y_test (numpy.ndarray): The true values of the target variable in the test set.
        - pred (numpy.ndarray): The predictions generated by the neural network for the test set.
    
    The function performs the following operations:
    1. Standardizes the features to the range [-1, 1] using MinMaxScaler.
    2. Splits the data into training, validation, and test sets.
    3. Creates a neural network model using the specified architecture.
    4. Trains the model using the training data and validates it with the validation set.
    5. Generates predictions on the test set and returns the true and predicted values.
    """

    # Assign 'df' to 'dataframe' to work with a new variable.
    self_dataframe = dataframe

    # Initialize a MinMaxScaler to standardize features to a range between -1 and 1.
    sc = MinMaxScaler(feature_range=(-1, 1))

    # Define the target variable, which is 'cooling_demand', for later use.
    self_target = target

    # List the features (excluding the target) that will be standardized.
    self_features = features

    num_features = len(features)

    # Iterate over each feature in the list.
    for var in self_features:
        # Standardize each feature to the range [-1, 1] using MinMaxScaler.
        # Reshape the data to be 2D, as required by the scaler.
        # Skip the target variable from standardization.
        if(var != self_target):
            self_dataframe[var] = sc.fit_transform(self_dataframe[var].values.reshape(-1, 1))

    # Convert the DataFrame to a NumPy array for model compatibility, removing labels.
    # First, drop the target column and convert the remaining features to a NumPy array.
    X = self_dataframe.drop(columns=self_target).to_numpy()

    # Convert the target variable to a NumPy array.
    Y = dataframe[self_target].to_numpy()

    # Set a random seed for reproducibility of the results.
    seed = 7
    np.random.seed(seed)

    # Split the dataset into training and testing sets.
    # 70% of the data is used for training, and 30% is used for testing.
    # This splits the data into X_train, X_test, y_train, and y_test.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

    # Further split the training set into training and validation sets.
    # 80% of the training data is used for training, and 20% is used for validation.
    # This helps in tuning hyperparameters and assessing model performance during training.
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    # Create a model using the 'create_model_large' function.
    # This function defines the architecture of the model, which is set to be large in this case.
    model = create_model_large(num_features, X_train)

    # Train the model using the training data and validate it with the validation data.
    # - 'X_train' and 'y_train' are the training features and labels respectively.
    # - 'epochs=150' specifies that the model will be trained for 150 epochs, or complete passes through the training dataset.
    # - 'batch_size=32' defines the number of samples processed before the model's weights are updated.
    # - 'validation_data=(X_val, y_val)' provides validation data to evaluate the model's performance after each epoch.
    #   This helps in monitoring the model's performance on unseen data and can help in early stopping to prevent overfitting.
    history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val))

    # Make predictions using the test data.
    # - 'model.predict(X_test)' generates predictions for the test set.
    # - '.reshape(1, -1)[0]' reshapes the prediction array to ensure it is a one-dimensional array, suitable for comparison with 'y_test'.
    pred = model.predict(X_test).reshape(1, -1)[0]
    
    return y_test, pred