import numpy as np
from matplotlib import pyplot as plt

def plot_targets(real, pred, targets, features, n_cols=2, figsize=(10, 10), percent=False):
    n_targets = len(targets)
    n_rows = int(np.ceil(n_targets / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten() if n_targets > 1 else [axs]

    for i, target in enumerate(targets):
        ax = axs[i]
        target_index = features.index(target)

        real_target = real[:, target_index]
        pred_target = pred[:, target_index]

        mae = np.abs(real_target - pred_target).mean()
        nmae = mae / (real_target.max() - real_target.min())
        nmse = ((real_target - pred_target) ** 2).mean() / np.var(real_target)
        nrmse = np.sqrt(((real_target - pred_target) ** 2).mean()) / (real_target.max() - real_target.min())

        if percent:
            nmae *= 100
            nmse *= 100
            nrmse *= 100
        
        ax.scatter(real_target, pred_target, alpha=0.6,
                   s=15, c=np.abs(real_target - pred_target), cmap='Spectral',
                   edgecolors='black',  linewidth=0.5,
                   label=f"NMAE={nmae:.3f}\nNMSE={nmse:.3f}\nNRMSE={nrmse:.3f}")
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

def plot_cmp(models, n_cols=2, figsize=(12, 10), percent=False):
    import numpy as np
    import matplotlib.pyplot as plt

    n_models = len(models)
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten() if n_models > 1 else [axs]

    for i, (ax, (model_name, (real, pred))) in enumerate(zip(axs, models.items())):
        # Assicura che real e pred siano monodimensionali
        real = np.ravel(real)
        pred = np.ravel(pred)
        errors = np.abs(real - pred)

        mae = errors.mean()
        nmae = mae / (real.max() - real.min())
        nmse = ((real - pred) ** 2).mean() / np.var(real)
        nrmse = np.sqrt(((real - pred) ** 2).mean()) / (real.max() - real.min())

        if percent:
            nmae *= 100
            nmse *= 100
            nrmse *= 100

        ax.scatter(real, pred, alpha=0.6, s=15, c=errors, cmap='Spectral',
                   edgecolors='black', linewidth=0.5,
                   label=f"NMAE={nmae:.3f}\nNMSE={nmse:.3f}\nNRMSE={nrmse:.3f}")
        ax.plot([real.min(), real.max()], [real.min(), real.max()], 'k-', linewidth=1)

        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(f"{model_name}")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend()

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle('Comparison of Model Predictions', fontsize=16)
    plt.tight_layout()
    plt.show()