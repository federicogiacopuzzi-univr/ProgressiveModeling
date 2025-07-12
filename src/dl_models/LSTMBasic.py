import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def create_supervised_sequences(data, timesteps):
    """
    Convert a 2D array into supervised LSTM input-output sequences.

    Parameters
    ----------
    - data (np.ndarray): Array of shape (samples, features)
    - timesteps (int): Number of time steps per input sequence

    Returns
    ----------
    - X (np.ndarray): Sequences of shape (num_samples, timesteps, features)
    - y (np.ndarray): Corresponding next-step targets (num_samples, features)
    """
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps])
    return np.array(X), np.array(y)

def create_lstm_model(timesteps, features, hidden_units=64):
    """
    Build and compile a basic LSTM model for multivariate time series forecasting.

    Architecture:
    LSTM → Dense (output one timestep of all features)

    Parameters
    ----------
    - timesteps (int): Length of input sequences
    - features (int): Number of features per timestep
    - hidden_units (int): Number of LSTM units

    Returns
    ----------
    - model (tf.keras.Model): Compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=(timesteps, features), activation='tanh'))
    model.add(Dense(features))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

def lstm_forecast(features,
                  dataframe=None,
                  train_df=None,
                  test_df=None,
                  timesteps=10,
                  hidden_units=64,
                  epochs=50,
                  batch_size=32,
                  test_size=0.2,
                  random_state=42):
    """
    Trains an LSTM on multivariate time series and predicts on the test set.
    
    Accepts either:
    - full `dataframe` to auto-split, or
    - manual `train_df` and `test_df`.
    
    Parameters
    ----------
    - features (list): Feature column names to use
    - dataframe/train_df/test_df: Data input (use only one method)
    - timesteps (int): Number of time steps for each input sequence
    - hidden_units (int): LSTM units
    - epochs (int): Number of training epochs
    - batch_size (int): Batch size for training
    - test_size (float): Used only if `dataframe` is provided
    
    Returns
    ----------
    - X_test (np.ndarray): Test input sequences (scaled), shape (samples, timesteps, features)
    - y_test (np.ndarray): True target values for test sequences (scaled)
    - y_pred (np.ndarray): Predicted target values from the model (scaled)
    - model (tf.keras.Model): Trained LSTM model
    """

    if dataframe is not None:
        scaler = MinMaxScaler()
        dataframe[features] = scaler.fit_transform(dataframe[features])
        data = dataframe[features].values
        split_index = int(len(data) * (1 - test_size))
        train_data, test_data = data[:split_index], data[split_index:]
    elif train_df is not None and test_df is not None:
        scaler = MinMaxScaler()
        train_df[features] = scaler.fit_transform(train_df[features])
        test_df[features] = scaler.transform(test_df[features])
        train_data = train_df[features].values
        test_data = test_df[features].values
    else:
        raise ValueError("Provide either `dataframe` or both `train_df` and `test_df`.")

    X_train, y_train = create_supervised_sequences(train_data, timesteps)
    X_test, y_test = create_supervised_sequences(test_data, timesteps)

    model = create_lstm_model(timesteps, len(features), hidden_units)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    y_pred = model.predict(X_test)

    return X_test, y_test, y_pred, model