import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def save_embeddings_plot(embeddings: np.array, labels: np.array, path: Path):
    """
    Saves a plot of the embeddings to the given path.
    Embeddings are colored according to the labels.
    If the embeddings hav more than 2 dimensions, they will be reduced to 2 dimensions using TSNE.

    Arguments:
        embeddings -- array of embeddings to plot with shape (n_users, n_features)
        labels -- array of labels with shape (n_users,) where 0 is genuine and 1 is fraud
        path -- path to save the plot to
    """
    users, features = np.shape(embeddings)

    if features > 2:
        print("Reduced dimensionality with TSNE")
        tsne = TSNE(n_components=2)
        embeddings = TSNE.fit_transform(embeddings)

    mask = labels == 0

    # plot the embeddings
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=mask, cmap='RdYlGn')

    # Add legend
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8),
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8)],
                labels=['Genuine', 'Fraud'])

    plt.savefig(path)
  