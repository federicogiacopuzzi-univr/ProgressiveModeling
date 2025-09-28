import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def plot_cmp(models, n_cols=2, figsize=(12, 10), percent=False, filename=None):
    """
    Compare multiple models by plotting their real vs predicted values with error metrics.

    Parameters:
    - models: dictionary where keys are model names and values are tuples (real, pred).
    - n_cols: number of subplot columns (default=2).
    - figsize: size of the figure (default=(12, 10)).
    - percent: if True, error metrics shown as percentage (default=False).
    """
    
    n_models = len(models)
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten() if n_models > 1 else [axs]

    for i, (ax, (model_name, (real, pred))) in enumerate(zip(axs, models.items())):
        # Flatten arrays
        real = np.ravel(real)
        pred = np.ravel(pred)

        # Scale between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        real_scaled = scaler.fit_transform(real.reshape(-1, 1)).flatten()
        pred_scaled = scaler.transform(pred.reshape(-1, 1)).flatten()

        # Absolute errors
        errors = np.abs(real_scaled - pred_scaled)

        # Calculate normalized error metrics
        mae = errors.mean()
        std = errors.std()
        nmae = mae
        nmse = ((real_scaled - pred_scaled) ** 2).mean() / np.var(real_scaled)
        nrmse = np.sqrt(((real_scaled - pred_scaled) ** 2).mean())
        nstd = std

        if percent:
            nmae *= 100
            nmse *= 100
            nrmse *= 100
            nstd *= 100

        # Scatter plot
        ax.scatter(real_scaled, pred_scaled, alpha=0.6, s=15, c=errors, cmap='Spectral',
                   edgecolors='black', linewidth=0.5,
                   label=f"NMAE={nmae:.3f}\nNMSE={nmse:.3f}\nNRMSE={nrmse:.3f}\nNSTD={nstd:.3f}")

        # Ideal reference line y=x
        ax.plot([0, 1], [0, 1], 'k-', linewidth=1)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(model_name)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend()

    # Remove unused axes
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle('Comparison of Model Predictions', fontsize=16)
    plt.tight_layout()

    # Save figure if filename is provided
    if filename is not None:
        full_path = "../reports/figures/" + filename
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_targets(real, pred, targets, features, n_cols=2, percent=False, filename=None):
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
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend()

    # Remove any unused subplots if targets < total subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()

    # Save figure if filename is provided
    if filename is not None:
        full_path = "../reports/figures/" + filename
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
def plot_aelstm_targets(real, pred, targets, features, n_cols=2, percent=False, filename=None):
    """
    Plot scatter plots of real vs predicted values (ultimo timestep) per target feature.
    For outputs in LSTM 3D shape (samples, timesteps, features).

    Parameters
    - real : 3D array containing the true values (samples, timesteps, features).
    - pred : 3D array containing the predicted values from the model (samples, timesteps, features).
    - targets : List of target feature names to plot.
    - features : List of all feature names present in the array (same order as the channels).
    - n_cols : Number of columns to use in the grid layout of plots.
    - percent : If True, the normalized error metrics (NMAE, NMSE, NRMSE, NSTD) will be shown as percentages.
    """

    n_targets = len(targets)
    n_rows = int(np.ceil(n_targets / n_cols))
    figsize = (5 * n_cols, 5 * n_rows)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten() if n_targets > 1 else [axs]

    for i, target in enumerate(targets):
        ax = axs[i]
        target_index = features.index(target)

        # Just extract the last timestep for each sequence
        real_target = real[:, -1, target_index]   # ultimo timestep
        pred_target = pred[:, -1, target_index]

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
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend()

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()

    # Save figure if filename is provided
    if filename is not None:
        full_path = "../reports/figures/" + filename
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_pca_variance(cumulative_var, n_components, filename=None):
    """
    Plot the cumulative explained variance of PCA components.

    Parameters:
    - cumulative_var: Cumulative sum of explained variance ratios from PCA.
    - n_components: The number of PCA components selected to reach a desired explained variance threshold.
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

    # Save figure if filename is provided
    if filename is not None:
        full_path = "../reports/figures/" + filename
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_importance(features, importances, stds, title="Feature Importance", filename=None):
        """
        Plots a horizontal bar chart of feature importance for autoencoders.

        Parameters
        ----------
        - features: List of feature names.
        - importances: Mean reconstruction error increase per feature.
        - stds: Standard deviations of the importance scores.
        - title: Plot title.
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
    
        # Save figure if filename is provided
        if filename is not None:
            full_path = "../reports/figures/" + filename
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            
        plt.show()

def plot_rfecv(rfecv, filename=None):
    """
    Plot RFECV results using MSE scores.

    Parameters
    rfecv: Fitted RFECV object.
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

    # Save figure if filename is provided
    if filename is not None:
        full_path = "../reports/figures/" + filename
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
    
    plt.show()