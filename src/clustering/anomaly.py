import numpy as np
from functools import reduce
from typing import Optional

from src.dataloader import BasicDataset
from src.similarity import UserSimilarity

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
        self.adj = dataset.graph_u2i.toarray().astype(int)
        self.user_simi = UserSimilarity(dataset.graph_u2i)

        if self.use_metadata:
            avg_ratings = dataset.metadata_df.groupby(dataset.METADATA_ITEM_ID)[dataset.METADATA_STAR_RATING].mean()
            self.average_ratings = avg_ratings.values
            self.rating_matrix = dataset.rated_graph_u2i.toarray()
            first_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].min()
            last_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].max()
            self.review_periods = (last_date-first_date).astype('timedelta64[D]')

        self.burstness_threshold = burstness_threshold

    def group_set_union(self, P_g_bit):
        """Generate the union of all products in the group
           Products that were rated by any user in the group will be 1, otherwise 0
        """
        union = None
        for p in P_g_bit:
            union = union | p
    
    def group_set_intersection(self, P_g_bit):
        """Generate the intersection of all products in the group
           Products that were rated by all users in the group will be 1, otherwise 0
        """
        intersection = None
        for p in P_g_bit:
            intersection = intersection & p

    def group_review_count(self, P_g_bit):
        """Generate the total number of reviews of all user in the group
        """
        sum = 0
        for p in P_g_bit:
            sum += p.count()
        return sum

    def penalty_function(self, R_g, P_g_bit):
        """
        Generates penalty function for the anomaly scores given a list of reviewers and products that they reviewed.
        Penalizes smaller groups because they are less likely to be a fake review farm.

        INPUTS:
        R_g (arr) - users in the group
        P_g_bit   - product sets in the group

        OUTPUTS:
        L_g (float) - penalty for the group
        """
        R_gnorm = len(R_g)
        P_gnorm = self.group_set_union(P_g_bit).sum()
        L_g = 1 / (1 + np.e**(-1 * (R_gnorm + P_gnorm - 3)))
        return L_g

    def review_tightness(self, R_g, P_g_bit):
        """
        Generates review tightness for the anomaly scores given a list of reviewers and products that they reviewed

        INPUTS:
        R_g (arr) - users in the group
        P_g_bit   - product sets in the group

        OUTPUTS:
        RT_g (float) - review tightness for the group
        """
        R_gnorm = len(R_g)
        P_gnorm = self.group_set_union(P_g_bit).count()
        L_g = self.penalty_function(R_g, P_g_bit)
        RT_g = (self.group_review_count(P_g_bit) * L_g) / (R_gnorm * P_gnorm)
        return RT_g


    def product_tightness(self, P_g_bit):
        """
        Generates product tightness for the anomaly scores given a list of reviewers and products that they reviewed

        INPUTS:
        P_g_bit - product sets in the group

        OUTPUTS:
        PT_g (float) - product tightness for the group
        """
        union_count = self.group_set_union(P_g_bit).count()
        intersection_count = self.group_set_intersection(P_g_bit).count()
        PT_g = 0 if union_count == 0 else intersection_count / union_count
        return PT_g
    
    def sum_jaccard(self, R_g):
        """
        Compute Jaccard similarity between all pairs of users and sum the scores
        """
        users = R_g
        if users <= 1:
            return 0
    
        similarity_scores = [
            self.user_simi.get_jaccard_similarity(R_g[i], R_g[j])
            for i in range(users)
            for j in range(i + 1, users)
        ]
        
        return sum(similarity_scores)


    def neighbor_tightness(self, R_g, P_g_bit):
        """
        Generates product tightness for the anomaly scores given a list of reviewers and products that they reviewed

        INPUTS:
        R_g (arr) - users in the group
        P_g_bit   - product sets in the group

        OUTPUTS:
        
        NT_g (float) - product tightness for the group
        """
        R_gnorm = len(R_g)
        js = self.sum_jaccard(R_g)
        L_g = self.penalty_function(R_g, P_g_bit)
        NT_g = (2 * js * L_g) / R_gnorm
        return NT_g

    def AVRD(self, R_g, P_g):
        """
        Generates average user rating deviation through the use of the dataset and matrix manipulation
        """
        R_gnorm = len(R_g)
        average_ratings_matrix = np.tile(self.average_ratings, (R_gnorm, 1))
        A_g = P_g * average_ratings_matrix
        ones = np.ones(self.adj.shape[1])
        AVRD_g = (np.abs(A_g - self.rating_matrix[R_g]) @ ones) / (P_g @ ones)
        return AVRD_g

    def BST(self, review_periods_g):
        """Generate burstness for all users in a group based off of times that the reviews were made"""
        BST_g = np.where(review_periods_g < self.burstness_threshold, 1 - review_periods_g / self.burstness_threshold, 0)
        return BST_g
    
    def generate_single_anomaly_score(self, R_g, P_g, P_g_bit, review_periods_g: Optional[np.ndarray] = None):
        """Generate single anomaly score for a cluster.
        
        INPUTS:
        R_g (arr) - users in the group
        P_g (arr) - product sets in the group as numpy array of 0s and 1s
                    shape (num_users_in_group, num_products_in_entire_dataset)
        P_g_bit   - list of product sets in the group as bitarrays of 0s and 1s
                    list of length num_users_in_group with bitarrays of length num_products_in_entire_dataset
        review_periods_g (arr) - period between the first and last review for the users in the group
        """
        R_gnorm = len(R_g)
        group_anomaly_compactness = self.review_tightness(R_g, P_g_bit) * self.product_tightness(P_g_bit) * self.neighbor_tightness(R_g, P_g_bit)

        if self.use_metadata:
            AVRD_g = self.AVRD(R_g, P_g)
            BST_g = self.BST(review_periods_g)
            anomaly_score = 3 * group_anomaly_compactness + np.sum(AVRD_g)/R_gnorm + np.sum(BST_g)/R_gnorm
        else:
            anomaly_score = 3 * group_anomaly_compactness

        return anomaly_score
    

    def generate_anomaly_scores(self):
        """Generate all anomaly scores for all clusters"""
        anomaly_scores = list()
        for R_g in self.clusters:
            P_g_bit = self.user_simi.get_user_bitarrays(R_g)
            P_g = self.adj[R_g]
            if self.use_metadata:
                score = self.generate_single_anomaly_score(R_g, P_g, P_g_bit, self.review_periods[R_g])
            else:
                score = self.generate_single_anomaly_score(R_g, P_g_bit)
            anomaly_scores.append(score)
        return anomaly_scores

