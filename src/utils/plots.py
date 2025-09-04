import numpy as np
from matplotlib import pyplot as plt

def plot_targets(real, pred, targets, features, n_cols=2, percent=False):
    """
    Plot scatter plots of real vs predicted values for specified target features.
    Displays error metrics (NMAE, NMSE, NRMSE, NSTD) as legend on each subplot.

    Parameters:
    - real: numpy array of true values.
    - pred: numpy array of predicted values.
    - targets: list of target feature names to plot.
    - features: list of all feature names in order.
    - n_cols: number of subplot columns (default=2).
    - percent: if True, error metrics are expressed in percentage (default=False).
    """

    n_targets = len(targets)
    n_rows = int(np.ceil(n_targets / n_cols))  # Calculate number of rows needed

    figsize = (5 * n_cols, 5 * n_rows)  # Figure size scales with number of plots

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    # Flatten axes array for easy iteration; handle case with single plot
    axs = axs.flatten() if n_targets > 1 else [axs]

    for i, target in enumerate(targets):
        ax = axs[i]
        target_index = features.index(target)  # Get column index of target feature

        real_target = real[:, target_index]  # Extract real values for the target
        pred_target = pred[:, target_index]  # Extract predicted values for the target

        errors = np.abs(real_target - pred_target)  # Absolute errors per sample

        mae = errors.mean()  # Mean Absolute Error
        std = errors.std()   # Standard deviation of errors
        # Normalized errors by the range of the true values
        nmae = mae / (real_target.max() - real_target.min())
        nmse = ((real_target - pred_target) ** 2).mean() / np.var(real_target)
        nrmse = np.sqrt(((real_target - pred_target) ** 2).mean()) / (real_target.max() - real_target.min())
        nstd = std / (real_target.max() - real_target.min())

        # Convert to percentage if requested
        if percent:
            nmae *= 100
            nmse *= 100
            nrmse *= 100
            nstd *= 100
        
        # Scatter plot: color mapped to absolute error for visualization
        ax.scatter(real_target, pred_target, alpha=0.6,
                   s=15, c=errors, cmap='Spectral',
                   edgecolors='black', linewidth=0.5,
                   label=f"NMAE={nmae:.3f}\nNMSE={nmse:.3f}\nNRMSE={nrmse:.3f}\nNSTD={nstd:.3f}")
        # Plot the ideal y=x line for reference
        ax.plot([real_target.min(), real_target.max()],
                [real_target.min(), real_target.max()],
                'k-', linewidth=1)
        ax.set_title(f"Target: {target}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend()

    # Remove any unused subplots if targets < total subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def plot_cmp(models, n_cols=2, figsize=(12, 10), percent=False):
    """
    Compare multiple models by plotting their real vs predicted values with error metrics.

    Parameters:
    - models: dictionary where keys are model names and values are tuples (real, pred).
    - n_cols: number of subplot columns (default=2).
    - figsize: size of the figure (default=(12, 10)).
    - percent: if True, error metrics shown as percentage (default=False).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    n_models = len(models)
    n_rows = int(np.ceil(n_models / n_cols))  # Compute rows needed

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten() if n_models > 1 else [axs]

    for i, (ax, (model_name, (real, pred))) in enumerate(zip(axs, models.items())):
        # Flatten arrays in case they're multi-dimensional
        real = np.ravel(real)
        pred = np.ravel(pred)
        errors = np.abs(real - pred)  # Absolute errors per sample

        mae = errors.mean()
        std = errors.std()
        nmae = mae / (real.max() - real.min())
        nmse = ((real - pred) ** 2).mean() / np.var(real)
        nrmse = np.sqrt(((real - pred) ** 2).mean()) / (real.max() - real.min())
        nstd = std / (real.max() - real.min())

        if percent:
            nmae *= 100
            nmse *= 100
            nrmse *= 100
            nstd *= 100

        # Scatter plot colored by error magnitude
        ax.scatter(real, pred, alpha=0.6, s=15, c=errors, cmap='Spectral',
                   edgecolors='black', linewidth=0.5,
                   label=f"NMAE={nmae:.3f}\nNMSE={nmse:.3f}\nNRMSE={nrmse:.3f}\nNSTD={nstd:.3f}")
        # Plot ideal reference line y=x
        ax.plot([real.min(), real.max()], [real.min(), real.max()], 'k-', linewidth=1)

        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(f"{model_name}")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend()

    # Remove unused axes if any
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle('Comparison of Model Predictions', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_aelstm_targets(real, pred, targets, features, n_cols=2, percent=False):
    """
    Plot scatter plots of real vs predicted values (ultimo timestep) per target feature.
    For outputs in LSTM 3D shape (samples, timesteps, features).
    """

    n_targets = len(targets)
    n_rows = int(np.ceil(n_targets / n_cols))
    figsize = (5 * n_cols, 5 * n_rows)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten() if n_targets > 1 else [axs]

    for i, target in enumerate(targets):
        ax = axs[i]
        target_index = features.index(target)

        # Estrai solo l'ultimo timestep per ogni sequenza
        real_target = real[:, :, target_index].mean(axis=1)
        pred_target = pred[:, :, target_index].mean(axis=1)


        errors = np.abs(real_target - pred_target)

        mae = errors.mean()
        std = errors.std()
        range_ = real_target.max() - real_target.min()
        var_ = np.var(real_target)

        nmae = mae / range_ if range_ != 0 else 0
        nmse = ((real_target - pred_target) ** 2).mean() / var_ if var_ != 0 else 0
        nrmse = np.sqrt(((real_target - pred_target) ** 2).mean()) / range_ if range_ != 0 else 0
        nstd = std / range_ if range_ != 0 else 0

        if percent:
            nmae *= 100
            nmse *= 100
            nrmse *= 100
            nstd *= 100

        ax.scatter(real_target, pred_target, alpha=0.6, s=15, c=errors, cmap='Spectral',
                   edgecolors='black', linewidth=0.5,
                   label=f"NMAE={nmae:.3f}\nNMSE={nmse:.3f}\nNRMSE={nrmse:.3f}\nNSTD={nstd:.3f}")
        ax.plot([real_target.min(), real_target.max()],
                [real_target.min(), real_target.max()],
                'k-', linewidth=1)
        ax.set_title(f"Target: {target}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend()

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def plot_pca_variance(cumulative_var, n_components):
    """
    Plot the cumulative explained variance of PCA components.

    Parameters:
    - cumulative_var: array-like
        Cumulative sum of explained variance ratios from PCA.
    - n_components: int
        The number of PCA components selected to reach a desired explained variance threshold.

    The function displays:
    - A line plot showing how the cumulative explained variance increases with the number of components.
    - A horizontal line indicating the variance threshold (e.g., 97%).
    - A vertical line showing the number of components needed to reach that threshold.
    """
    
    plt.figure(figsize=(12, 10))
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o', linestyle='--', color='b')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Components Number')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.axhline(y=0.97, color='r', linestyle='--', label='90% variance')
    plt.axvline(x=n_components, color='g', linestyle='--', label=f'{n_components} components')
    plt.legend()
    plt.legend()
    plt.show()

def plot_feature_importance(features, importances, stds, title="Feature Importance"):
        """
        Plots a horizontal bar chart of feature importance for autoencoders.

        Parameters
        ----------
        - features (list or array-like): List of feature names.
        - importances (np.ndarray): Mean reconstruction error increase per feature.
        - stds (np.ndarray): Standard deviations of the importance scores.
        - title (str): Plot title.
        """
        # Normalize importances and stds
        max_importance = np.max(importances)
        importances[importances < 0] = 0
        importances = importances / max_importance
        stds = stds / max_importance
    
        sorted_idx = np.argsort(importances)

        plt.figure(figsize=(10, 6))
        plt.barh(np.array(features)[sorted_idx], importances[sorted_idx], xerr=stds[sorted_idx])
        plt.xlabel("Increase in Reconstruction Error")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_rfecv(rfecv):
    """
    Plot RFECV results using MSE scores.

    Parameters
    ----------
    rfecv : RFECV
        Fitted RFECV object.
    """
    if hasattr(rfecv, "cv_results_"):
        scores = rfecv.cv_results_['mean_test_score']
    elif hasattr(rfecv, "grid_scores_"):
        scores = rfecv.grid_scores_
    else:
        raise AttributeError("RFECV object has no grid_scores_ or cv_results_")

    n_features = range(1, len(scores) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(n_features, scores, marker='o')
    plt.xlabel("Number of features selected")
    plt.ylabel("MSE (negative)")
    plt.title("RFECV Feature Selection")
    plt.grid(True)
    plt.show()