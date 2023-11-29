import numpy as np
from src.dataloader import BasicDataset
from src.similarity import UserSimilarity
from bitarray import bitarray

class AnomalyGroup:

    def __init__(self, 
                 users: list, 
                 group_product_set_intersection: bitarray, 
                 group_product_set_union: bitarray, 
                 n_total_reviews: int, 
                 user_simi: UserSimilarity, 
                 child1: 'AnomalyGroup'= None, 
                 child2: 'AnomalyGroup' = None):
        self.users = users
        self.group_product_set_intersection = group_product_set_intersection
        self.group_product_set_union = group_product_set_union
        self.n_total_products_reviewed = group_product_set_union.count()
        self.n_common_products_reviewed = group_product_set_intersection.count()
        self.n_users = len(users)
        self.n_total_reviews = n_total_reviews
        self.user_simi = user_simi
        self.child1 = child1
        self.child2 = child2

    @staticmethod
    def make_group_from_children(child1: 'AnomalyGroup', child2: 'AnomalyGroup', user_simi: UserSimilarity):
        users = child1.users + child2.users
        group_product_set_intersection = child1.group_product_set_intersection & child2.group_product_set_intersection
        group_product_set_union = child1.group_product_set_union | child2.group_product_set_union
        n_total_reviews = child1.n_total_reviews + child2.n_total_reviews

        group = AnomalyGroup(
            users=users,
            group_product_set_intersection=group_product_set_intersection,
            group_product_set_union=group_product_set_union,
            n_total_reviews=n_total_reviews,
            user_simi=user_simi,
            child1=child1,
            child2=child2
        )

        return group
    
    @staticmethod
    def make_single_user_group(self, user: int, user_simi: UserSimilarity):
        users = [user]
        group_product_set_intersection = user_simi.get_user_bitarray(user)
        group_product_set_union = user_simi.get_user_bitarray(user)
        n_total_reviews = group_product_set_union.count()

        group = AnomalyGroup(
            users=users,
            group_product_set_intersection=group_product_set_intersection,
            group_product_set_union=group_product_set_union,
            n_total_reviews=n_total_reviews,
            user_simi=user_simi
        )

        return group

    def _penalty_function(self):
        """
        Generates penalty function a group of users.
        Penalizes smaller groups because they are less likely to be a fake review farm.

        OUTPUTS:
            L_g (float) - penalty for the group
        """
        self.penalty = 1 / (1 + np.e**(-1 * (self.n_users + self.n_total_products_reviewed - 3)))
        return self.penalty

    def _review_tightness(self):
        """
        Generates review tightness for a group.

        OUTPUTS:
            RT_g (float) - review tightness for the group
        """
        self.review_tightness = (self.n_total_reviews * self.penalty) / (self.n_users * self.n_total_products_reviewed)
        return self.review_tightness

    def _product_tightness(self):
        """
        Generates product tightness for a group of users.

        OUTPUTS:
            PT_g (float) - product tightness for the group
        """
        if self.n_total_products_reviewed == 0:
           return 0
        self.product_tightness = self.n_common_products_reviewed / self.n_total_products_reviewed
        return self.product_tightness
    
    def _average_jaccard(self):
        """
        Compute average Jaccard similarity between all pairs of users in a group.
        If there is only 0 or 1 users in the group, this returns 1.

        OUTPUTS:
            average_jaccard (float) - average jaccard similarity between all pairs of users
        """
        if self.n_users <= 1:
            self.average_jaccard = 1
            return 1
        
        similarity_scores_children = [
            self.user_simi.get_jaccard_similarity(i, j)
            for i in self.child1.users
            for j in self.child2.users
        ]

        child1_similarity_score_sum = self.child1.average_jaccard * self.child1.n_users
        child2_similarity_score_sum = self.child2.average_jaccard * self.child2.n_users

        self.average_jaccard = (2 * similarity_scores_children + child1_similarity_score_sum + child2_similarity_score_sum) / self.n_users
        
        return self.average_jaccard

    def _neighbor_tightness(self):
        """
        Generates neighbor tightness for a group of users.

        OUTPUTS:
            NT_g (float) - neighbor tightness for the group
        """
        NT_g = self._average_jaccard() * self.penalty
        return NT_g
    
    def group_anomaly_compactness(self):
        """
        Generates group anomaly compactness for a group of users.

        INPUTS:
            group (arr) - list of users in the group

        OUTPUTS:
            Pi_g (float) - group anomaly compactness for the group
        """
        self._penalty_function()
        RT_g = self._review_tightness()
        PT_g = self._product_tightness()
        NT_g = self._neighbor_tightness()
        Pi_g = 3 * RT_g * PT_g * NT_g
        return Pi_g    

def get_overall_anomaly_score(users: list, group_anomaly_compactness: float, use_metadata: bool, avrd: np.array = None, burstness: np.array = None):
    if use_metadata:
        group_mean_avrd = np.mean(avrd[users])
        group_mean_burstness = np.mean(burstness[users])
        score = 3 * group_anomaly_compactness + group_mean_avrd + group_mean_burstness
    else:
        score = group_anomaly_compactness
    return score


def hierarchical_anomaly_scores(linkage_matrix, dataset: BasicDataset, use_metadata: bool = True, burstness_threshold: int = 0.5):
    """
    Generates anomaly scores for each group in the hierarchical clustering linkage matrix.

    INPUTS:
        linkage_matrix: linkage matrix from scipy hierarchical clustering
        dataset: BasicDataset object
        use_metadata: boolean indicating whether to use metadata
        burstness_threshold: threshold for burstness

    OUTPUTS:
        groups: list of AnomalyGroup objects
        anomaly_scores: list of anomaly scores for each group in the linkage matrix
    """

    user_simi = UserSimilarity(dataset.graph_u2i)

    if use_metadata:
        avg_ratings = dataset.metadata_df.groupby(dataset.METADATA_ITEM_ID)[dataset.METADATA_STAR_RATING].mean()
        average_ratings = avg_ratings.values
        rating_matrix = dataset.rated_graph_u2i.toarray()

        average_ratings_matrix = np.tile(average_ratings, (dataset.n_users, 1))
        A_g = dataset.graph_u2i * average_ratings_matrix
        avrd_mat = np.abs(A_g - rating_matrix)
        avrd = np.sum(avrd_mat, axis=0)


        first_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].min()
        last_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].max()
        review_periods = (last_date-first_date).astype('timedelta64[D]')
        burstness = np.where(review_periods < burstness_threshold, 1 - review_periods / burstness_threshold, 0)

    groups = []
    anomaly_scores = []

    for user in range(dataset.n_users):
        group = AnomalyGroup.make_single_user_group(user, user_simi)
        score = get_overall_anomaly_score(group.users, group.group_anomaly_compactness(), use_metadata, avrd, burstness)
        anomaly_scores.append(score)
        groups.append(group)
 
    for row in linkage_matrix:
        child1 = row[0]
        child2 = row[1]
        group = AnomalyGroup(groups[child1], groups[child2])
        score = get_overall_anomaly_score(group.users, group.group_anomaly_compactness(), use_metadata, avrd, burstness)
        anomaly_scores.append(score)
        groups.append(group)

    return groups, anomaly_scores



        