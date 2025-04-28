import time
import subprocess
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics

def create_autoencoder_model(input_dim, latent_dim, neurons_per_layer):
    r"""
    Creates and compiles an autoencoder model with dynamic encoder and decoder layers, 
    using the specified number of neurons and latent dimension.
    
    Parameters
    ----------
    - input_dim (int): The number of input features.
    - latent_dim (int): The dimension of the latent (bottleneck) space.
    - neurons_per_layer (list): A list of integers specifying the number of neurons 
      in each hidden layer (for both encoder and decoder).
    
    Returns
    -------
    - model (tensorflow.keras.Model): A Keras Model object representing the autoencoder.

    The function performs the following steps:
    1. Creates an input layer with the specified input dimension.
    2. Constructs the encoder part of the model by adding dense layers with the specified 
       number of neurons and ReLU activation.
    3. Adds a latent space (bottleneck) layer that represents the reduced dimensionality.
    4. Constructs the decoder part of the model by reversing the encoder layers.
    5. Adds the output layer to reconstruct the input from the latent space, with sigmoid activation.
    6. Compiles the model using the Adam optimizer and mean squared error loss function, 
       with Mean Absolute Error (MAE) as a metric.
    """
    
    # Define the input layer using Input(...)
    input_layer = Input(shape=(input_dim,))
    
    # Encoder: Dynamically construct encoder layers based on the number of neurons provided
    x = input_layer
    for neurons in neurons_per_layer:
        x = layers.Dense(neurons, activation='relu')(x)  # Add a dense layer with 'relu' activation
    
    # Bottleneck (latent space): The layer that compresses the input into a lower-dimensional space
    latent_space = layers.Dense(latent_dim, activation='relu')(x)  # Add the latent space layer
    
    # Decoder: Dynamically construct decoder layers (reverse the neurons for symmetry)
    for neurons in reversed(neurons_per_layer):
        latent_space = layers.Dense(neurons, activation='relu')(latent_space)  # Add a dense layer to decode
    
    # Output layer: The final layer that reconstructs the input from the latent space
    output_layer = layers.Dense(input_dim, activation='sigmoid')(latent_space)  # Output layer (reconstruction)
    
    # Create the model by specifying the input and output layers
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model with Adam optimizer, Mean Squared Error (MSE) loss, and Mean Absolute Error (MAE) as a metric
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=[metrics.mae])
    
    return model

def grid_search_autoencoder(
    dataframe,
    features,
    latent_dims=[10],
    batch_sizes=[32],
    neurons_per_layer=[64],
    epochs=100,
    test_size=0.2,
    random_state=42,
    verbose=1):
    r"""
    Performs a grid search on an autoencoder by testing different hyperparameter combinations.

    Parameters
    ----------
    - dataframe (pandas.DataFrame): The input data.
    - features (list): List of feature names to use in the autoencoder.
    - latent_dims (list, optional): List of latent dimension sizes to try.
    - batch_sizes (list, optional): List of batch sizes to try.
    - neurons_per_layer (list, optional): List of neurons per hidden layer (optional).
    - epochs (int, optional): Number of epochs to train for each configuration.
    - test_size (float, optional): Fraction of data to use for validation (default is 0.2).
    - random_state (int, optional): Random seed for reproducibility (default is 42).
    - verbose (int, optional): Verbosity mode for model training (0 = silent, 1 = progress bar).

    Returns
    -------
    - pandas.DataFrame: A DataFrame containing all hyperparameter combinations and their final validation loss.

    The function performs the following steps:
    1. Scales the input features to the range [-1, 1] using MinMaxScaler.
    2. Splits the data into training and validation sets.
    3. Creates combinations of hyperparameters from the provided lists.
    4. For each combination, creates and trains the autoencoder model.
    5. Records the final validation loss after training.
    6. Returns a DataFrame with the hyperparameter combinations and corresponding validation loss.
    """
    
    # Scale the features to range [-1, 1]
    df = dataframe.copy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for var in features:
        df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))
    
    X = df[features].to_numpy()

    # Split the data into training and validation
    X_train, X_val = train_test_split(X, test_size=test_size, random_state=random_state)

    # Create all combinations of hyperparameters
    combinations = list(itertools.product(latent_dims, batch_sizes, neurons_per_layer))

    results = []

    start_time = time.time()

    print("Starting Grid Search...")

    for idx, (latent_dim, batch_size, neurons) in enumerate(combinations):
        # Create a new model
        model = create_autoencoder_model(input_dim=X_train.shape[1], latent_dim=latent_dim, neurons_per_layer=neurons_per_layer)

        # Train the model
        history = model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            verbose=verbose
        )

        # Record final validation loss
        val_loss = history.history['val_loss'][-1]

        results.append({
            'latent_dim': latent_dim,
            'batch_size': batch_size,
            'val_loss': val_loss
        })

    total_time = time.time() - start_time
    print(f"\nTotal Grid Search execution time: {total_time:.2f} seconds")

    return pd.DataFrame(results)