class GraphSimilarity:
    """Class for computing similarity between 2 nodes.
    Replaces large np array of pairwise similarities that may not be able to fit in memory.
    """

    def __init__(self, graph_u2u):
        self.graph_u2u = graph_u2u
        self.n = graph_u2u.shape[0]

    def __getitem__(self, index):
        """Get similarity between node indices [x, y]"""
        x, y = index
        column_vector = self.graph_u2u.getcol(x)
        row_vector = self.graph_u2u.getrow(y)
        sim = column_vector.dot(row_vector)
        return sim
