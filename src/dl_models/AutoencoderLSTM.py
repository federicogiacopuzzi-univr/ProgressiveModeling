from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import numpy as np

def create_lstm_autoencoder_model(timesteps, features, latent_dim):
    """
    Creates and compiles an LSTM autoencoder model for sequence reconstruction.

    Parameters
    ----------
    - timesteps (int): Number of time steps in each input sequence.
    - features (int): Number of features per time step.
    - latent_dim (int): Dimensionality of the latent space (encoding dimension).

    Returns
    ----------
    - model (tensorflow.keras.Model): Compiled LSTM autoencoder model.
    """
    model = Sequential()
    # Encoder LSTM
    model.add(LSTM(latent_dim, activation='relu', input_shape=(timesteps, features)))
    # Repeat the latent vector 'timesteps' times for the decoder input
    model.add(RepeatVector(timesteps))
    # Decoder LSTM returning sequences
    model.add(LSTM(latent_dim, activation='relu', return_sequences=True))
    # TimeDistributed dense layer to reconstruct features at each time step
    model.add(TimeDistributed(Dense(features)))

    optimizer = Adam(0.00007)
    
    model.compile(optimizer='adam', loss='mse')
    
    return model

def create_sequences(data, timesteps):
    """
    Converts a 2D array (samples x features) into 3D sequences for LSTM input.
    
    Parameters
    ----------
    - data (numpy.ndarray): 2D array of shape (samples, features)
    - timesteps (int): Number of time steps per sequence
    
    Returns
    ----------
    - sequences (numpy.ndarray): 3D array of shape (num_sequences, timesteps, features)
    """
    sequences = []
    for i in range(len(data) - timesteps + 1):
        sequences.append(data[i:i+timesteps])
    return np.array(sequences)

def Autoencoder(features,
                     dataframe=None,
                     train_df=None,
                     test_df=None,
                     epochs=100,
                     latent_dim=32,
                     test_size=0.3,
                     random_state=7,
                     batch_size=32,
                     timesteps=10):
    """
    Creates and trains an LSTM autoencoder model for sequence data reconstruction.

    Steps:
    1. Scales features to [0,1].
    2. Splits data into train and test if not provided.
    3. Converts data into sequences for LSTM input.
    4. Builds LSTM autoencoder and trains it.
    5. Returns test sequences and reconstructed sequences.

    Parameters
    ----------
    - features (list): Feature column names.
    - dataframe/train_df/test_df (pandas.DataFrame): Data input.
    - epochs, latent_dim, batch_size, etc: Training parameters.
    - timesteps (int): Length of sequences for LSTM input.

    Returns
    ----------
    - X_test_seq (numpy.ndarray): Test sequences.
    - X_test_pred (numpy.ndarray): Model reconstruction on test sequences.
    """
    
    if dataframe is not None:
        scaler = MinMaxScaler(feature_range=(0,1))
        dataframe[features] = scaler.fit_transform(dataframe[features])
        X = dataframe[features].to_numpy()
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
    elif train_df is not None and test_df is not None:
        scaler = MinMaxScaler(feature_range=(0,1))
        train_df[features] = scaler.fit_transform(train_df[features])
        test_df[features] = scaler.transform(test_df[features])
        X_train = train_df[features].to_numpy()
        X_test = test_df[features].to_numpy()
    else:
        raise ValueError("Either dataframe or both train_df and test_df must be provided.")

    # Create sequences for LSTM
    X_train_seq = create_sequences(X_train, timesteps)
    X_test_seq = create_sequences(X_test, timesteps)

    # Create LSTM autoencoder model
    model = create_lstm_autoencoder_model(timesteps=timesteps, features=len(features), latent_dim=latent_dim)

    # Train the model to reconstruct sequences
    model.fit(X_train_seq, X_train_seq, epochs=epochs, batch_size=batch_size, verbose=1)

    # Predict reconstruction on test sequences
    X_test_pred = model.predict(X_test_seq)

    return X_test_seq, X_test_pred, model
