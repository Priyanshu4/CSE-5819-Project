from bitarray import bitarray
from scipy.sparse import csr_matrix

class UserSimilarity:
    """
    Class for computing similarities between 2 user nodes.
    Replaces large np array of pairwise similarities that may not be able to fit in memory.
    Takes user to item sparse matrix as input. Internally represents users as bitarrays.
    Similarities are computed lazily and efficiently when queried.
    """

    def __init__(self, graph_u2i: csr_matrix):
        self._graph_u2i = graph_u2i
        self._n_users = graph_u2i.shape[0]
        self.shape = (self._n_users, self._n_users)
        self._init_bitarrays()

    def _init_bitarrays(self):
        self._user_bitarrays = []

        for row in range(self._n_users):
            user_vector = self._graph_u2i.getrow(row)
            user_bitarray = self._user_vector_to_bitarray(user_vector)
            self._user_bitarrays.append(user_bitarray)

    def _user_vector_to_bitarray(self, user_vector):
        """Converts a user vector in sparse format (row vector) to a bitarray."""
        non_zero_row_indices, non_zero_column_indices = user_vector.nonzero()
        bit_array = bitarray(user_vector.shape[1])
        bit_array.setall(0)
        for i in non_zero_column_indices:
            bit_array[i] = 1
        return bit_array
    
    def get_user_bitarray(self, user_index):
        return self._user_bitarrays[user_index]
    
    def get_user_bitarrays(self, user_indices):
        return [self._user_bitarrays[i] for i in user_indices]
    
    def get_jaccard_similarity(self, u1: int, u2: int):
        """
        Gets the jaccard similarity between user1 and user2 using their indices.
        """
        u1 = self.get_user_bitarray(u1)
        u2 = self.get_user_bitarray(u2)
        intersection = u1 & u2
        union = u1 | u2
        union_count = union.count()
        intersection_count = intersection.count()

        if union_count == 0:
            return 1
        
        return intersection_count / union_count

    def get_smoothed_jaccard_similarity(self, u1: int , u2: int):
        """
        Gets the smoothed jaccard similarity between user1 and user2.
        """
        u1 = self.get_user_bitarray(u1)
        u2 = self.get_user_bitarray(u2)
        intersection = u1 & u2
        union = u1 | u2
        union_count = union.count()
        intersection_count = intersection.count()

        if intersection_count == 0:
            simi_score = 1 / (union_count + len(u1))
        elif intersection_count == union_count:
            simi_score = (union_count + len(u1) - 1) / (union_count + len(u1))
        else:
            simi_score = intersection_count / union_count
        return float(simi_score)
    
    def items_in_common(self, u1: int, u2: int):
        """ 
        Returns the number of items in common between user1 and user2.
        In other words, the size of the intersection of their product sets.
        """
        u1 = self.get_user_bitarray(u1)
        u2 = self.get_user_bitarray(u2)
        intersection = u1 & u2
        items_in_common = intersection.count()
        return items_in_common
