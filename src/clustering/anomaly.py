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
    def make_group_from_children(child1: 'AnomalyGroup', child2: 'AnomalyGroup', user_simi: UserSimilarity) -> 'AnomalyGroup':
        """
        Makes an anomaly group object from two child anomaly groups.

        INPUTS:
            child1 (AnomalyGroup) - AnomalyGroup object
            child2 (AnomalyGroup) - AnomalyGroup object
            user_simi (UserSimilarity) - UserSimilarity object
        
        OUTPUTS:
            group (AnomalyGroup) - AnomalyGroup object
        """
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
    def make_single_user_group(user: int, user_simi: UserSimilarity) -> 'AnomalyGroup':
        """
        Makes an anomaly group object for a single user.

        INPUTS:
            user - the user in the group
            user_simi (UserSimilarity) - UserSimilarity object

        OUTPUTS:    
            group (AnomalyGroup) - AnomalyGroup object
        """
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

    
    @staticmethod
    def make_group(users: list, user_simi: UserSimilarity) -> 'AnomalyGroup':
        """
        Makes an anomaly group object from a list of users.

        INPUTS:
            users (list) - list of users in the group
            user_simi (UserSimilarity) - UserSimilarity object

        OUTPUTS:    
            group (AnomalyGroup) - AnomalyGroup object
        """
        user_bitarrays = user_simi.get_user_bitarrays(users)
        group_product_set_intersection = user_bitarrays[0]  
        group_product_set_union = user_bitarrays[0]
        n_total_reviews = user_bitarrays[0].count()
        for i in range(1, len(user_bitarrays)):
            group_product_set_intersection = group_product_set_intersection & user_bitarrays[i]
            group_product_set_union = group_product_set_union | user_bitarrays[i]
            n_total_reviews += user_bitarrays[i].count()

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
        RT_g = (self.n_total_reviews * self.penalty) / (self.n_users * self.n_total_products_reviewed)
        return RT_g

    def _product_tightness(self):
        """
        Generates product tightness for a group of users.

        OUTPUTS:
            PT_g (float) - product tightness for the group
        """
        if self.n_total_products_reviewed == 0:
           return 0
        PT_g = self.n_common_products_reviewed / self.n_total_products_reviewed
        return PT_g
    
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

        self.average_jaccard = (2 * sum(similarity_scores_children) + child1_similarity_score_sum + child2_similarity_score_sum) / self.n_users
        
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

def get_overall_anomaly_score(group: AnomalyGroup, use_metadata: bool, avrd: np.array = None, burstness: np.array = None):
    """
    Generates overall anomaly score for a group of users.
    3 * group_anomaly_compactness + group_mean_avrd + group_mean_burstness

    INPUTS:
        group: AnomalyGroup object
        use_metadata (bool): boolean indicating whether to use metadata (avrd and burstness)
        avrd (array): average rating deviation for all users in dataset
        burstness (array): burstness for all users in dataset

    OUTPUTS:
        score (float): overall anomaly score for the group
    """
    if use_metadata:
        group_mean_avrd = np.mean(avrd[group.users])
        group_mean_burstness = np.mean(burstness[group.users])
        score = 3 * group.group_anomaly_compactness() + group_mean_avrd + group_mean_burstness
    else:
        score = 3 * group.group_anomaly_compactness()
    return score


def hierarchical_anomaly_scores(linkage_matrix, dataset: BasicDataset, use_metadata: bool = True, burstness_threshold: int = 0.5):
    """
    Generates anomaly scores for each group in the hierarchical clustering linkage matrix.

    INPUTS:
        linkage_matrix: linkage matrix from scipy hierarchical clustering
        dataset: BasicDataset object
        use_metadata: boolean indicating whether to use metadata
        burstness_threshold: threshold for burstness in days

    OUTPUTS:
        groups: list of AnomalyGroup objects
        anomaly_scores: list of anomaly scores for each group in the linkage matrix
    """

    user_simi = UserSimilarity(dataset.graph_u2i)

    if use_metadata:
        avg_ratings = dataset.metadata_df.groupby(dataset.METADATA_ITEM_ID)[dataset.METADATA_STAR_RATING].mean()
        product_average_ratings = avg_ratings.values 
        graph_u2i = dataset.graph_u2i.toarray()
        rating_matrix = dataset.rated_graph_u2i.toarray()
        average_ratings_matrix = np.tile(product_average_ratings, (dataset.n_users, 1))
        diff_matrix = (rating_matrix - average_ratings_matrix) * graph_u2i
        sum_of_diffs = np.sum(np.abs(diff_matrix), axis=1)
        num_rated_products = np.sum(graph_u2i, axis=1)
        avrd = np.where(num_rated_products != 0, sum_of_diffs / num_rated_products, 0)


        first_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].min()
        last_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].max()
        review_periods = (last_date-first_date).dt.days
        burstness = np.where(review_periods < burstness_threshold, 1 - review_periods / burstness_threshold, 0)

    else:
        avrd = None
        burstness = None

    groups = []
    anomaly_scores = np.zeros(dataset.n_users + len(linkage_matrix), dtype=float)

    for user in range(dataset.n_users):
        group = AnomalyGroup.make_single_user_group(user, user_simi)
        score = get_overall_anomaly_score(group, use_metadata, avrd, burstness)
        anomaly_scores[user] = score
        groups.append(group)
 
    for i, row in enumerate(linkage_matrix):
        child1 = int(row[0])
        child2 = int(row[1])
        group = AnomalyGroup.make_group_from_children(groups[child1], groups[child2], user_simi)
        score = get_overall_anomaly_score(group, use_metadata, avrd, burstness)
        anomaly_scores[i + dataset.n_users] = score
        groups.append(group)

    return groups, anomaly_scores



        