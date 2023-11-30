from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

class HClust:
    def __init__(self, data):
        self.data = data
        self.Z = None

    def generate_linkage_matrix(self, method='average'):
        """
        Generates linkage matrix from given data

        Inputs:
        method (str) - linkage method to generate the linkage matrix
                       see scipy.cluster.hierarchy.linkage for more info

        Outputs:
        Z (ndarr) - linkage matrix generated from initialized data
        """

        self.Z = linkage(self.data, method=method)
        return self.Z
    

    def generate_dendrogram(self,filepath=None):
        """
        Generates dendrogram and saves it to the specified file path

        Inputs:
        filepath (str) - file path to store the dendrogram png

        Outputs:
        1 - if successfully outputted dendrogram
        """

        fig = plt.figure(figsize=(8, 6))
        ax = fig.gca()
        dendrogram_info = dendrogram(self.Z, ax=ax)
        if filepath:
            fig.savefig(filepath+'/dendrogram.png', dpi=300)

        return 1


    def find_all_leaves(self):
        """
        Finds all leaves under every branch with the generated linkage matrix 

        Outputs:
        leaves (list) - list of lists of all leaves under each branch
        """

        leaves = list()

        for row in self.Z:
            branch = []
            self._find_leaves_iterative(row, branch)
            leaves.append(branch)
        
        return leaves

    
    def _find_leaves_iterative(self, row, leaves=None):
        """
        Iterative helper function for hclust to find all leaves under a specific branch

        Inputs:
        row (ndarray) - row of the linkage matrix
        leaves (list) - list to add the nodes to

        Outputs:
        leaves (list) - list of nodes under a branch

        """

        if leaves is None:
            leaves = []

        maxlen = int(self.Z[-1, -1]) - 1
        stack = [(row, 0)]

        while stack:
            current_row, current_depth = stack.pop()

            lefti = int(current_row[0])
            righti = int(current_row[1])

            if lefti <= maxlen:
                leaves.append(lefti)
            else:
                newrow = self.Z[lefti - maxlen-1]
                stack.append((newrow, current_depth + 1))

            if righti <= maxlen:
                leaves.append(righti)
            else:
                newrow = self.Z[righti - maxlen-1]
                stack.append((newrow, current_depth + 1))

        return leaves
