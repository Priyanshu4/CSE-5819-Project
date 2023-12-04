import numpy as np
import pandas as pd
from src.dataloader import BasicDataset
from src.similarity import UserSimilarity
from bitarray import bitarray

import src.utils as utils


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

        self.penalty = None
        self.average_jaccard = None

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
    def make_group_from_many_children(children: list['AnomalyGroup'], user_simi: UserSimilarity) -> 'AnomalyGroup':
        """
        Makes an anomaly group object from many child anomaly groups.
        Internally uses make_group_from_children() to merge two groups at a time.
        Recursively merges two halves of the list of children until there is only one group left.

        INPUTS:
            children (list[AnomalyGroup]) - list of AnomalyGroup objects
            user_simi (UserSimilarity) - UserSimilarity object
        
        OUTPUTS:
            group (AnomalyGroup) - AnomalyGroup object
        """
        if len(children) == 1:
            return children[0]
        elif len(children) == 2:
            return AnomalyGroup.make_group_from_children(children[0], children[1], user_simi)
        else:
            return AnomalyGroup.make_group_from_children(
                AnomalyGroup.make_group_from_many_children(children[:len(children) // 2], user_simi),
                AnomalyGroup.make_group_from_many_children(children[len(children) // 2:], user_simi),
                user_simi
            )
        
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

        if self.child1.average_jaccard is None:
            self.child1._average_jaccard()
        if self.child2.average_jaccard is None:
            self.child2._average_jaccard()

        child1_similarity_score_sum = self.child1.average_jaccard * self.child1.n_users**2
        child2_similarity_score_sum = self.child2.average_jaccard * self.child2.n_users**2

        self.average_jaccard = (2 * sum(similarity_scores_children) + child1_similarity_score_sum + child2_similarity_score_sum) / self.n_users**2
        
        return self.average_jaccard

    def _neighbor_tightness(self):
        """
        Generates neighbor tightness for a group of users.

        OUTPUTS:
            NT_g (float) - neighbor tightness for the group
        """
        NT_g = self._average_jaccard() * self.penalty
        return NT_g
    
    def group_anomaly_compactness(self, enable_penalty: bool = False):
        """
        Generates group anomaly compactness for a group of users.

        INPUTS:
            group (arr) - list of users in the group
            enable_penalty (bool) - boolean indicating whether to enable penalty for smaller groups

        OUTPUTS:
            Pi_g (float) - group anomaly compactness for the group
        """
        if enable_penalty:
            self._penalty_function()
        else:
            self.penalty = 1
        RT_g = self._review_tightness()
        PT_g = self._product_tightness()
        NT_g = self._neighbor_tightness()
        Pi_g = AnomalyScorer.weighted_geometric_mean(np.array([RT_g, PT_g, NT_g]), np.array([1/3, 1/3, 1/3]))
        return Pi_g    

class AnomalyScorer:

    def __init__(self, dataset: BasicDataset, enable_penalty: bool, use_metadata: bool = True, burstness_threshold: int = 30):
        """
        Initializes AnomalyScorer object.
        
        INPUTS:
            dataset (BasicDataset) - BasicDataset object
            enable_penalty (bool) - boolean indicating whether to enable penalty for smaller groups
            use_metadata (bool) - boolean indicating whether to use metadata for anomaly scoring
            burstness_threshold (int) - minimum number of days user must be active for to have 0 burstness score
        """
        self.dataset = dataset
        self.enable_penalty = enable_penalty
        self.use_metadata = use_metadata
        self.burstness_threshold = burstness_threshold
        self.user_simi = UserSimilarity(dataset.graph_u2i)

        if use_metadata:
            self.avg_ratings = dataset.metadata_df.groupby(dataset.METADATA_ITEM_ID)[dataset.METADATA_STAR_RATING].mean()
            self.product_average_ratings = self.avg_ratings.values 
            self.graph_u2i = dataset.graph_u2i.toarray()
            self.rating_matrix = dataset.rated_graph_u2i.toarray()
            self.average_ratings_matrix = np.tile(self.product_average_ratings, (dataset.n_users, 1))
            self.diff_matrix = (self.rating_matrix - self.average_ratings_matrix) * self.graph_u2i
            self.sum_of_diffs = np.sum(np.abs(self.diff_matrix), axis=1)
            self.num_rated_products = np.sum(self.graph_u2i, axis=1)
            self.avrd = np.where(self.num_rated_products != 0, self.sum_of_diffs / self.num_rated_products, 0)

            self.first_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].min()
            self.last_date = dataset.metadata_df.groupby(dataset.METADATA_USER_ID)[dataset.METADATA_DATE].max()
            self.review_periods = (self.last_date-self.first_date).dt.days
            self.burstness = np.where(self.review_periods < self.burstness_threshold, 1 - self.review_periods / self.burstness_threshold, 0)

        else:
            self.avrd = None
            self.burstness = None

    def get_anomaly_score(self, group: list | AnomalyGroup):
        """
        Generates overall anomaly score for a group of users.

        INPUTS:
            group: AnomalyGroup object

        OUTPUTS:
            score (float): overall anomaly score for the group
        """
        if isinstance(group, list):
            group = AnomalyGroup.make_group(group, self.user_simi) 
        if self.use_metadata:
            group_mean_avrd = np.mean(self.avrd[group.users]) / 5 # divide by 5 to normalize (max 5 star rating difference)
            group_mean_burstness = np.mean(self.burstness[group.users])
            score = AnomalyScorer.weighted_geometric_mean(
                np.array([group.group_anomaly_compactness(self.enable_penalty), group_mean_avrd, group_mean_burstness]), 
                np.array([4/5, 1/10, 1/10]))
        else:
            score = group.group_anomaly_compactness(self.enable_penalty)
        return score


    def hierarchical_anomaly_scores(self, linkage_matrix, group_mapping: dict = None):
        """
        Generates anomaly scores for each group in the hierarchical clustering linkage matrix.

        INPUTS:
            linkage_matrix: linkage matrix from scipy hierarchical clustering
            group_mapping: dictionary mapping indices in the linkage matrix to indices in the original dataset

        OUTPUTS:
            groups: list of AnomalyGroup objects
            anomaly_scores: list of anomaly scores for each group in the linkage matrix
        """
        if group_mapping is None:
            group_mapping = utils.IdentityMap()
            n_users = self.dataset.n_users
        else:
            n_users = len(group_mapping.keys())

        groups = []
        anomaly_scores = np.zeros(n_users + len(linkage_matrix), dtype=float)

        for i in range(n_users):
            user = group_mapping[i]
            group = AnomalyGroup.make_single_user_group(user, self.user_simi)
            score = self.get_anomaly_score(group)
            anomaly_scores[i] = score
            groups.append(group)
    
        for i, row in enumerate(linkage_matrix):
            child1 = int(row[0])
            child2 = int(row[1])
            group = AnomalyGroup.make_group_from_children(groups[child1], groups[child2], self.user_simi)
            score = self.get_anomaly_score(group)
            anomaly_scores[i + n_users] = score
            groups.append(group)
            
        return groups, anomaly_scores
    
    def hdbscan_tree_anomaly_scores(self, condensed_tree_df: pd.DataFrame):
        """
        Generates anomaly scores for each group in the HDBSCAN condensed tree.
        
        INPUTS:
            condensed_tree_df: condensed tree from HDBSCAN as pandas dataframe.
                               See https://hdbscan.readthedocs.io/en/latest/api.html

        OUTPUTS:
            groups: list of AnomalyGroup objects
            anomaly_scores: np array of anomaly scores for each group in the condensed tree            
        """
        # Group dataframe by parent column and sort parents by indices.
        with utils.timer(name="sorting and grouping"):
            parent_groups = condensed_tree_df.sort_values('parent', ascending=False).groupby('parent', sort=False)

        # Preallocate groups and anomaly_scores to size of number of groups
        with utils.timer(name="preallocating"):
            ngroups = len(parent_groups) + self.dataset.n_users
            groups = [None] * ngroups
            anomaly_scores = np.zeros(ngroups, dtype=float)

        # Initialize single user groups
        with utils.timer(name="initializing single user groups"):
            for user in range(self.dataset.n_users):
                group = AnomalyGroup.make_single_user_group(user, self.user_simi)
                score = self.get_anomaly_score(group)
                anomaly_scores[user] = score
                groups[user] = group

        # Iterate through parent groups
        for parent, group in parent_groups:

            with utils.timer(name="aggregating children"):
                children = group['child'].values
                child_groups = [groups[child] for child in children]
                if None in child_groups:
                    raise RuntimeError("condensed_tree_df is not in the expected format. Please check the documentation for the condensed_tree_df argument.\n" +
                                    "We expect the condensed_tree_df to contain a row for each parent-child pair.\n" +
                                    "The parent column should not contain any values less than dataset.n_users. Parent at dataset.n_users is the root of all nodes.\n")
                
            with utils.timer(name="making group"):
                group = AnomalyGroup.make_group_from_many_children(child_groups, self.user_simi)

            with utils.timer(name="computing score"):
                score = self.get_anomaly_score(group)
                anomaly_scores[parent] = score
                groups[parent] = group

        print(utils.timer.formatted_tape_str(select_keys=["sorting and grouping", "preallocating", "initializing single user groups", "aggregating children", "making group", "computing score"]))

        return groups, anomaly_scores


    @staticmethod
    def weighted_geometric_mean(scores: np.ndarray, weights: np.ndarray):
        """
        Computes the weighted geometric mean of the scores.
        """
        if np.any(scores == 0.0):
            return 0
        return np.exp(AnomalyScorer.weighted_arithmetic_mean(np.log(scores), weights))

    @staticmethod
    def weighted_arithmetic_mean(scores: np.ndarray, weights: np.ndarray):
        """
        Computes the weighted arithmetic mean of the scores.
        """
        return np.sum(scores * weights) / np.sum(weights)