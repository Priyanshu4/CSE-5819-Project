import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_loss_epochs(loss_values: np.ndarray, path: Path):
    """
    Plot the loss values across epochs and save to the given path.

    Parameters:
    loss_values (np.array): An array of loss values for each epoch.
    """
    epochs = np.arange(len(loss_values))
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs, y=loss_values)
    plt.title("Loss Across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()