import numpy as np
from sklearn.utils import shuffle

def importance_autoencoder(model, X, features, n_repeats=10):
    """
    Compute permutation feature importance for a base autoencoder (non-sequential).

    Parameters
    ----------
    - model (tf.keras.Model): Trained autoencoder model.
    - X (np.ndarray): Input data of shape (samples, features).
    - features (list): List of feature names.
    - n_repeats (int): Number of permutation repeats per feature.

    Returns
    -------
    - importance_means (np.ndarray): Mean increase in reconstruction error per feature.
    - importance_stds (np.ndarray): Std deviation of increases.
    """
    
    baseline_preds = model.predict(X)
    baseline_error = np.mean((X - baseline_preds) ** 2)

    n_features = X.shape[1]
    importances = np.zeros((n_repeats, n_features))

    for f_idx in range(n_features):
        for n in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, f_idx] = shuffle(X[:, f_idx], random_state=n)

            preds = model.predict(X_permuted)
            error = np.mean((X_permuted - preds) ** 2)
            importances[n, f_idx] = error - baseline_error

    importance_means = np.mean(importances, axis=0)
    importance_stds = np.std(importances, axis=0)

    return importance_means, importance_stds

def importance_lstm(model, X, y_true, features, n_repeats=10):
    """
    Compute permutation feature importance for an LSTM forecasting model.

    Parameters
    ----------
    - model (tf.keras.Model): Trained LSTM model.
    - X (np.ndarray): Input sequences, shape (samples, timesteps, features).
    - y_true (np.ndarray): True target values, shape (samples, features).
    - features (list): List of feature names.
    - n_repeats (int): Number of permutation repeats per feature.

    Returns
    ----------
    - importance_means (np.ndarray): Mean increase in prediction error per feature.
    - importance_stds (np.ndarray): Std deviation of error increase per feature.
    """
    
    baseline_pred = model.predict(X)
    baseline_error = np.mean((y_true - baseline_pred) ** 2)

    n_features = X.shape[2]
    importances = np.zeros((n_repeats, n_features))

    for f_idx in range(n_features):
        for n in range(n_repeats):
            X_permuted = np.copy(X)
            # Permute the selected feature across all samples and timesteps
            for t in range(X.shape[1]):
                X_permuted[:, t, f_idx] = shuffle(X[:, t, f_idx], random_state=n)
            
            pred = model.predict(X_permuted)
            error = np.mean((y_true - pred) ** 2)
            importances[n, f_idx] = error - baseline_error

    importance_means = np.mean(importances, axis=0)
    importance_stds = np.std(importances, axis=0)

    return importance_means, importance_stds

def importance_autoencoder_lstm(model, X_seq, features, n_repeats=10):
    """
    Compute permutation feature importance for an autoencoder model by measuring
    the increase in reconstruction error when permuting each feature.

    Parameters
    ----------
    - model (tf.keras.Model): Trained autoencoder model.
    - X_seq (np.ndarray): Input sequences of shape (samples, timesteps, features).
    - features (list): List of feature names.
    - n_repeats (int): Number of times to repeat permutation per feature.

    Returns
    -------
    - importance_means (np.ndarray): Mean increase in reconstruction error per feature.
    - importance_stds (np.ndarray): Standard deviation of the increases.
    """
    
    # Baseline reconstruction error
    baseline_preds = model.predict(X_seq)
    baseline_error = np.mean((X_seq - baseline_preds) ** 2)

    n_features = X_seq.shape[2]
    importances = np.zeros((n_repeats, n_features))

    for f_idx in range(n_features):
        for n in range(n_repeats):
            X_permuted = np.copy(X_seq)
            # Permute values of the feature f_idx across all samples and timesteps
            for t in range(X_seq.shape[1]):
                X_permuted[:, t, f_idx] = shuffle(X_seq[:, t, f_idx], random_state=n)

            preds = model.predict(X_permuted)
            error = np.mean((X_permuted - preds) ** 2)
            importances[n, f_idx] = error - baseline_error

    importance_means = np.mean(importances, axis=0)
    importance_stds = np.std(importances, axis=0)

    return importance_means, importance_stds