from similarity import *
import numpy as np

class AnomalyScore:
    def __init__(self, leaves, mapping, adj):
        self.leaves = leaves
        self.mapping = mapping
        self.adj = adj

    def penalty_function(self, reviewers, products):
        """
        Generates penalty function for the anomaly scores given a list of reviewers and products that they reviewed

        INPUTS:
        group (list) - list of users 

        """

        R_g = len(reviewers)
        P_g = len(products)
        L_g = 1 / (1 + np.e**(-1 * (R_g + P_g - 3)))

        return L_g

    def review_tightness(self):
        pass

    def product_tightness(self):
        pass

    def neighbor_tightness(self):
        pass
    