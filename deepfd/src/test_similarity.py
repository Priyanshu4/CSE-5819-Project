import unittest
import numpy as np
from scipy.sparse import csr_matrix
from similarity import GraphSimilarity  

def get_simi(u1, u2):
    nz_u1 = u1.nonzero()[0]
    nz_u2 = u2.nonzero()[0]
    nz_inter = np.array(list(set(nz_u1) & set(nz_u2)))
    nz_union = np.array(list(set(nz_u1) | set(nz_u2)))
    if len(nz_inter) == 0:
        simi_score = 1 / (len(nz_union) + len(u1))
    elif len(nz_inter) == len(nz_union):
        simi_score = (len(nz_union) + len(u1) - 1) / (len(nz_union) + len(u1))
    else:
        simi_score = len(nz_inter) / len(nz_union)
    return float(simi_score)

class TestGraphSimilarity(unittest.TestCase):

    def setUp(self):
        # Create test matrices 
        self.test_matrices = [
            np.array([[1, 0], [0, 1]]),  # 2x2 matrix
            np.array([[1, 1], [1, 0]]),  # Another 2x2 matrix
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # 3x3 Identity matrix
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),  # 3x3 matrix with all 1s
        ]

        # Convert test matrices to sparse format
        self.sparse_matrices = [csr_matrix(matrix) for matrix in self.test_matrices]

        # Create instances of GraphSimilarity for each sparse matrix
        self.graph_sim_instances = [GraphSimilarity(matrix) for matrix in self.sparse_matrices]

    def test_shape(self):
        # Test if the shape is correctly set for each matrix
        for i, matrix in enumerate(self.test_matrices):
            with self.subTest(i=i):
                self.assertEqual(self.graph_sim_instances[i].shape, matrix.shape)

    def test_similarity(self):
        # Test the similarity calculation for each pair of nodes in each matrix
        for i, sparse_matrix in enumerate(self.sparse_matrices):
            with self.subTest(i=i):
                for row1 in range(sparse_matrix.shape[0]):
                    for row2 in range(sparse_matrix.shape[0]):
                        expected_similarity = get_simi(sparse_matrix.getrow(row1).toarray()[0], sparse_matrix.getrow(row2).toarray()[0])
                        actual_similarity = self.graph_sim_instances[i][row1, row2]
                        self.assertAlmostEqual(expected_similarity, actual_similarity)

if __name__ == "__main__":
    unittest.main()