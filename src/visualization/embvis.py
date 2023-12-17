import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.manifold import TSNE

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

def plot_embeddings(embeddings: np.array, labels: np.array, path: Path = None, title=None):
    """
    Saves a plot of the embeddings to the given path. 
    Embeddings are colored according to the labels.
    If the embeddings have more than 2 dimensions, they will be reduced to 2 dimensions using TSNE.

    Arguments:
        embeddings -- array of embeddings to plot with shape (n_users, n_features)
        labels -- array of labels with shape (n_users,) where 0 is genuine and a positive integer is fraud
                  if there are only two labels (0 and 1), 0 is labeled as 'Genuine' and 1 as 'Fraudulent'
                  if there are more than two labels, 0 is 'Genuine' and each positive integer is a different 'Fraud Group'
        path -- path to save the plot to
        title -- title of the plot
    """
    users, features = np.shape(embeddings)

    if features > 2:
        tsne = TSNE(n_components=2)
        embeddings = tsne.fit_transform(embeddings)

    unique_labels = np.unique(labels)
    n_unique_labels = len(unique_labels)
  
    if n_unique_labels <= 2:
        pallete = colors.ListedColormap(["green", "red"])
    elif n_unique_labels <= 5:
        pallete = colors.ListedColormap(["green", "red", "blue", "orange", "purple"])
    elif n_unique_labels <= 10:
        pallete = plt.cm.get_cmap('tab10', n_unique_labels)
    elif n_unique_labels <= 20:
        pallete = plt.cm.get_cmap('tab20', n_unique_labels)
    else:
        raise NotImplementedError("Plotting embeddings with more than 20 seperate labels is not yet supported.")

    plt.figure(figsize=(10, 8))

    if n_unique_labels > 1:
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if n_unique_labels == 2 and label == 1:
                label_name = 'Fraudulent'
            elif label == 0:
                label_name = 'Genuine'
            else:
                label_name = f'Fraud Group {label}'
            
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], color=pallete(i / (n_unique_labels - 1)), label=label_name)
        plt.legend()
    else:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], color=pallete(0))

    plt.title('User Embeddings' if title is None else title)
    plt.xlabel('TSNE Component 1' if features > 2 else 'Latent Space Component 1')
    plt.ylabel('TSNE Component 2' if features > 2 else 'Latent Space Component 2')

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
