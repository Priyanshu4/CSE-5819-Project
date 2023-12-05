import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

def plot_embeddings(embeddings: np.array, labels: np.array, path: Path = None):
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
        tsne = TSNE(n_components=2)
        embeddings = tsne.fit_transform(embeddings)

    mask = labels == 0

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=mask, cmap='RdYlGn')

    plt.title('User Embeddings with True Labels')

    if features > 2:
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')
    else:
        plt.xlabel('Latent Space Component 1')
        plt.ylabel('Latent Space Component 2')   

    # Add legend
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8),
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8)],
                labels=['Genuine', 'Fraud'])

    
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def plot_embeddings_with_anomaly_scores(embeddings: np.array, user_anomaly_scores: np.array, path: Path = None):
    """
    Saves a plot of the embeddings to the given path. 
    Embeddings are colored according to the anomaly score of a user.
    Users which have a higher anomaly score are given a more red color, while lower anomaly scores have a more blue color.
    If the embeddings have more than 2 dimensions, they will be reduced to 2 dimensions using TSNE.

    Arguments:
        embeddings -- array of embeddings to plot with shape (n_users, n_features)
        user_anomaly_scores -- array of anomaly scores with shape (n_users,) and values between 0 and 1
        path -- path to save the plot to
    """    
    users, features = np.shape(embeddings)

    if features > 2:
        tsne = TSNE(n_components=2)
        embeddings = tsne.fit_transform(embeddings)

    scatter = plt.scatter(embeddings[:,0], embeddings[:,1], c=user_anomaly_scores, cmap='coolwarm')  

    # Add color bar
    plt.colorbar(scatter, label='Anomaly Scores')
    plt.title('User Embeddings with Anomaly Scores')

    if features > 2:
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')
    else:
        plt.xlabel('Latent Space Component 1')
        plt.ylabel('Latent Space Component 2')   

    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
