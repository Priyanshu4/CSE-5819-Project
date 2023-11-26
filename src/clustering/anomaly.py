import numpy as np
from functools import reduce

from src.dataloader import BasicDataset

class AnomalyScore:
    """ Anomaly scores for fake reviewer group detection implemented according to the following paper:
        https://arxiv.org/pdf/2112.06403.pdf
    """

    def __init__(self, clusters, dataset: BasicDataset, use_metadata = True, burstness_threshold=0):
        """ 
        INPUTS:
        clusters (arr) - list of clusters, where each cluster is a list of users
        dataset (BasicDataset) - dataset object
        use_metadata (bool) - whether to use metadata (review times and average ratings) or not
        burstness_threshold (int) - threshold for burstness
        """
        self.clusters = clusters
        self.use_metadata = use_metadata
        self.adj = dataset.graph_u2i.toarray()

        if self.use_metadata:
            avg_ratings = dataset.metadata_df.groupby(dataset.METADATA_ITEM_ID)[dataset.METADATA_STAR_RATING].mean()
            self.average_ratings = avg_ratings.values
            self.rating_matrix = dataset.rated_graph_u2i.toarray()
            first_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].min()
            last_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].max()
            self.review_times = (last_date-first_date).astype('timedelta64[D]')

        self.burstness_threshold = burstness_threshold

    def penalty_function(self, R_g, P_g):
        """
        Generates penalty function for the anomaly scores given a list of reviewers and products that they reviewed.
        Penalizes smaller groups because they are less likely to be a fake review farm.

        INPUTS:
        R_g (arr) - users in the group
        P_g (arr) - product sets in the group

        OUTPUTS:
        L_g (float) - penalty for the group
        """
        R_gnorm = len(R_g)
        P_gnorm = len(P_g)
        L_g = 1 / (1 + np.e**(-1 * (R_gnorm + P_gnorm - 3)))
        return L_g

    def review_tightness(self, R_g, P_g):
        """
        Generates review tightness for the anomaly scores given a list of reviewers and products that they reviewed

        INPUTS:
        R_g (arr) - users in the group
        P_g (arr) - product sets in the group

        OUTPUTS:
        RT_g (float) - review tightness for the group
        """
        R_gnorm = len(R_g)
        P_gnorm = len(P_g)
        L_g = self.penalty_function(R_g, P_g)
        RT_g = (np.sum(P_g) * L_g) / (R_gnorm * P_gnorm)
        return RT_g


    def product_tightness(self, P_g):
        """
        Generates product tightness for the anomaly scores given a list of reviewers and products that they reviewed

        INPUTS:
        P_g (arr) - ndarr of product sets in the group

        OUTPUTS:
        PT_g (float) - product tightness for the group
        """
        PT_g = len(reduce(np.intersect1d, P_g)) / len(reduce(np.union1d, P_g))
        return PT_g
    
    
    def jaccard_similarity(self, s1, s2):
        """
        Compute Jaccard similarity between two sets, represented as NumPy arrays.

        INPUTS:
        s1 (ndarr) - first array.
        s2 (ndarr) - second array.
        
        OUTPUTS:
        js -  Jaccard similarity coefficient.
        """
        intersection = np.intersect1d(s1, s2)
        union = np.union1d(s1, s2)
        
        # Handle zero division error if union is empty
        if len(union) == 0:
            return 0.0
        
        js = len(intersection) / len(union)
        return js
    
    def sum_jaccard(self, arrays):
        """
        Compute Jaccard similarity between all pairs and sum the scores
        """
        if len(arrays) <= 1:
            return 0
    
        similarity_scores = [
            self.jaccard_similarity(arrays[i], arrays[j])
            for i in range(len(arrays))
            for j in range(i + 1, len(arrays))
        ]
        
        return sum(similarity_scores)


    def neighbor_tightness(self, R_g, P_g):
        """
        Generates product tightness for the anomaly scores given a list of reviewers and products that they reviewed

        INPUTS:
        R_g (arr) - users in the group
        P_g (arr) - product sets in the group

        OUTPUTS:
        
        NT_g (float) - product tightness for the group
        """
        R_gnorm = len(R_g)
        js = self.sum_jaccard(P_g)
        L_g = self.penalty_function(R_g, P_g)
        NT_g = (2 * js * L_g) / R_gnorm
        return NT_g


    def AVRD(self, R_g, P_g):
        """
        Generates average user rating deviation through the use of the dataset and matrix manipulation
        """
        R_gnorm = len(R_g)
        P_gnorm = len(P_g)
        average_ratings_matrix = np.tile(self.average_ratings, (R_gnorm, 1))
        A_g = P_g * average_ratings_matrix
        ones = np.ones(self.adj.shape[1])
        AVRD_g = (np.abs(A_g - self.rating_matrix[R_g]) @ ones) / (P_g @ ones)
        return AVRD_g


    def BST(self, review_times_g):
        """Generate burstness based off of times that the reviews were made"""
        BST_g = np.where(review_times_g < self.burstness_threshold, 1 - review_times_g / self.burstness_threshold, 0)
        return BST_g
    
        
    def generate_single_anomaly_score(self, R_g, P_g, review_times_g):
        """Generate single anomaly score"""
        R_gnorm = len(R_g)
        group_anomaly_compactness = self.review_tightness(R_g,P_g) * self.product_tightness(P_g) * self.neighbor_tightness(R_g, P_g)
        AVRD_g = self.AVRD(R_g, P_g)
        BST_g = self.BST(review_times_g)
        anomaly_score = 3 * group_anomaly_compactness + np.sum(AVRD_g)/R_gnorm + np.sum(BST_g)/R_gnorm
        return anomaly_score
    

    def generate_anomaly_scores(self):
        """Generate all anomaly scores"""
        anomaly_scores = list()
        for R_g in self.clusters:
            P_g = self.adj[R_g]
            anomaly_scores.append(self.generate_single_anomaly_score(R_g, P_g, self.review_times[R_g]))
        
        return anomaly_scores

