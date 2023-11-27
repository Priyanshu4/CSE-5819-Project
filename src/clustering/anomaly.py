import numpy as np
from typing import Optional

from src.dataloader import BasicDataset
from src.similarity import UserSimilarity

class AnomalyScore:
    """ Anomaly scores for fake reviewer group detection implemented according to the following paper:
        https://arxiv.org/pdf/2112.06403.pdf
    """

    def __init__(self, dataset: BasicDataset, use_metadata = True, burstness_threshold=0):
        """ 
        INPUTS:
            dataset (BasicDataset) - dataset object
            use_metadata (bool) - whether to use metadata (review times and average ratings) or not
            burstness_threshold (int) - threshold for burstness
        """
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

    def group_product_set_union(self, P_g_bit):
        """Generate the union of all products in the group.
                
        INPUTS:
            P_g_bit - list of product sets in the group as bitarrays 
            
        OUTPUTS:
            union - union of all products in the group as bitarray
        """
        union = None
        for p in P_g_bit:
            union = union | p
        return union

    def total_products_reviewed_count(self, P_g_bit):
        """ Get the total number of products reviewed by any users in the group.
                
        INPUTS:
            P_g_bit - list of product sets in the group as bitarrays 
            
        OUTPUTS:
            total_products_reviewed - total number of products reviewed by any users in the group
        """
        total_products_reviewed = self.group_product_set_union(P_g_bit).count()
        return total_products_reviewed
       
    def group_product_set_intersection(self, P_g_bit):
        """Generate the intersection of all products in the group

        INPUTS:
            P_g_bit - list of product sets in the group as bitarrays        
        
        OUTPUTS:
            intersection - intersection of all products in the group as bitarray"""
        intersection = None
        for p in P_g_bit:
            intersection = intersection & p
        return intersection

    def common_products_reviewed_count(self, P_g_bit):
        """ Get the total number of products reviewed by all users in the group.
                
        INPUTS:
            P_g_bit - list of product sets in the group as bitarrays 
            
        OUTPUTS:
            common_products_reviewed - number of products reviewed by all users in the group
        """
        common_products_reviewed = self.group_product_set_intersection(P_g_bit).count()
        return common_products_reviewed
    
    def group_review_count(self, P_g_bit):
        """Get the total number of reviews of all user in the group.

        INPUTS:
            P_g_bit - list of product sets in the group as bitarrays
        
        OUTPUTS:
            reviews - total number of reviews of all user in the group
        """
        reviews = 0
        for p in P_g_bit:
            reviews += p.count()
        return reviews

    def penalty_function(self, n_users_in_group, n_total_products_reviewed):
        """
        Generates penalty function a group of users.
        Penalizes smaller groups because they are less likely to be a fake review farm.

        INPUTS:
            n_users_in_group (int) - number of users in the group
            n_total_products_reviewed (int) - total number of products reviewed by any users in the group

        OUTPUTS:
            L_g (float) - penalty for the group
        """
        L_g = 1 / (1 + np.e**(-1 * (n_users_in_group + n_total_products_reviewed - 3)))
        return L_g

    def review_tightness(self, n_users_in_group, n_total_reviews, n_total_products_reviewed, penalty):
        """
        Generates review tightness for a group.

        INPUTS:
            n_users_in_group (int) - number of users in the group
            n_total_reviews (int) - total number of reviews of all user in the group
            n_total_products_reviewed (int) - total number of products reviewed by any users in the group
            penalty (float) - size penalty for the group, L_g

        OUTPUTS:
            RT_g (float) - review tightness for the group
        """
        RT_g = (n_total_reviews * penalty) / (n_users_in_group * n_total_products_reviewed)
        return RT_g


    def product_tightness(self, n_total_products_reviewed, n_common_products_reviewed):
        """
        Generates product tightness for a group of users.

        INPUTS:
            total_products_reviewed_count (int) - total number of products reviewed by any users in the group
            common_products_reviewed_count (int) - number of products reviewed by all users in the group

        OUTPUTS:
            PT_g (float) - product tightness for the group
        """
        if n_total_products_reviewed == 0:
           return 0
        return n_common_products_reviewed / n_total_products_reviewed
    
    def average_jaccard(self, group):
        """
        Compute average Jaccard similarity between all pairs of users in a group.

        INPUTS:
            group (arr) - list of users in the group

        OUTPUTS:
            average_jaccard (float) - average jaccard similarity between all pairs of users
        """
        users = len(group)
        similarity_scores = [
            self.user_simi.get_jaccard_similarity(group[i], group[j])
            for i in range(users)
            for j in range(i + 1, users)
        ]
        
        return 2 * sum(similarity_scores) / users

    def neighbor_tightness(self, group, penalty):
        """
        Generates neighbor tightness for a group of users.

        INPUTS:
            group (arr) - list of users in the group
            penalty (float) - size penalty for the group, L_g

        OUTPUTS:
            NT_g (float) - neighbor tightness for the group
        """
        NT_g = self.average_jaccard(group) * penalty
        return NT_g
    
    def group_anomaly_compactness(self, group):
        """
        Generates group anomaly compactness for a group of users.

        INPUTS:
            group (arr) - list of users in the group

        OUTPUTS:
            Pi_g (float) - group anomaly compactness for the group
        """
        P_g_bit = self.user_simi.get_user_bitarrays(group)
        n_users_in_group = len(group)
        n_total_products_reviewed = self.total_products_reviewed_count(P_g_bit)
        n_common_products_reviewed = self.common_products_reviewed_count(P_g_bit)
        n_total_reviews = self.group_review_count(P_g_bit)
        penalty = self.penalty_function(n_users_in_group, n_total_products_reviewed)
        RT_g = self.review_tightness(n_users_in_group, n_total_reviews, n_total_products_reviewed, penalty)
        PT_g = self.product_tightness(n_total_products_reviewed, n_common_products_reviewed)
        NT_g = self.neighbor_tightness(group, penalty)
        Pi_g = 3 * RT_g * PT_g * NT_g
        return Pi_g

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
    
    def anomaly_score(self, group):
        """Generate anomaly score for a group of users.
        
        INPUTS:
        group (arr) - list of users in the group
        """
        Pi_g = self.group_anomaly_compactness(group)
        anomaly_score = 3 * Pi_g
        if self.use_metadata:
            P_g = self.adj[group]
            AVRD_g = self.AVRD(group, P_g)
            BST_g = self.BST(self.review_periods[group])
            anomaly_score = anomaly_score + np.mean(AVRD_g) + np.mean(BST_g)
        return anomaly_score    

    def generate_anomaly_scores(self, clusters):
        """ Generate anomaly scores for a group of clusters.
        
        OUTPUTS:
            anomaly_scores (arr) - list of anomaly scores for each cluster
        """
        anomaly_scores = np.zeros(len(clusters))
        for i, group in enumerate(clusters):
            anomaly_scores[i] = self.anomaly_score(group)
        return anomaly_scores

