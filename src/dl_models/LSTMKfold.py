import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
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
                  random_state=42,
                  kfold=1):
    """
    LSTM training and prediction with optional k-fold cross validation.

    Parameters
    ----------
    - features (list): feature column names
    - dataframe/train_df/test_df: input data; provide either dataframe or train_df+test_df
    - timesteps, hidden_units, epochs, batch_size: training params
    - test_size: used only if dataframe provided without train/test split
    - random_state: for reproducibility
    - kfold (int): number of folds for cross-validation (default=1 = no CV)

    Returns
    ----------
    - If kfold=1: (y_test, y_pred, scaler)
    - If kfold>1: (all_y_tests, all_y_preds) concatenated from folds
    """
    if dataframe is not None:
        scaler = MinMaxScaler()
        dataframe[features] = scaler.fit_transform(dataframe[features])
        data = dataframe[features].values
    elif train_df is not None and test_df is not None:
        scaler = MinMaxScaler()
        train_df[features] = scaler.fit_transform(train_df[features])
        test_df[features] = scaler.transform(test_df[features])
        data = np.concatenate([train_df[features].values, test_df[features].values])
    else:
        raise ValueError("Provide either dataframe or both train_df and test_df")

    if kfold > 1:
        kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
        all_y_tests = []
        all_y_preds = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):
            print(f"\nFold {fold}/{kfold}")

            train_data = data[train_idx]
            val_data = data[val_idx]

            X_train, y_train = create_supervised_sequences(train_data, timesteps)
            X_val, y_val = create_supervised_sequences(val_data, timesteps)

            if len(X_train) == 0 or len(X_val) == 0:
                print(f"Fold {fold} skipped: sequence length too short.")
                continue

            model = create_lstm_model(timesteps, len(features), hidden_units)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

            y_pred = model.predict(X_val)

            all_y_tests.append(y_val)
            all_y_preds.append(y_pred)

        return np.concatenate(all_y_tests, axis=0), np.concatenate(all_y_preds, axis=0)

    else:
        # No cross-validation: normal train/test split or from train_df/test_df
        if dataframe is not None:
            split_index = int(len(data) * (1 - test_size))
            train_data = data[:split_index]
            test_data = data[split_index:]
        else:
            train_data = train_df[features].values
            test_data = test_df[features].values

        X_train, y_train = create_supervised_sequences(train_data, timesteps)
        X_test, y_test = create_supervised_sequences(test_data, timesteps)

        model = create_lstm_model(timesteps, len(features), hidden_units)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

        y_pred = model.predict(X_test)

        return y_test, y_pred