import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Generate some multidimensional data (3D in this case)
data = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6], [6, 7, 8]])

# Calculate pairwise distances using Euclidean distance
distances = np.linalg.norm(data[:, np.newaxis] - data, axis=-1)

# Use the distances with the linkage function
linkage_matrix = linkage(distances, method='average')

# You can still create a dendrogram, but you may not visualize it directly
# Instead, you can plot the hierarchical clustering result as a tree
dendrogram(linkage_matrix, no_plot=True)

# Plot the hierarchical clustering tree
plt.figure(figsize=(10, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
dendrogram(linkage_matrix, labels=range(len(data)), leaf_rotation=90)
plt.show()
