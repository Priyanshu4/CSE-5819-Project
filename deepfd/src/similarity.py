from bitarray import bitarray


class GraphSimilarity:
    """Class for computing similarity between 2 user nodes.
    Replaces large np array of pairwise similarities that may not be able to fit in memory.
    Takes user to item sparse matrix as input.
    """

    def __init__(self, graph_u2i):
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

    def __getitem__(self, index):
        """Get similarity between user node indices [x, y]"""
        x, y = index
        user_x = self._user_bitarrays[x]
        user_y = self._user_bitarrays[y]
        return self._get_simi_bitarray(user_x, user_y)

    def _get_simi_bitarray(self, u1, u2):
        """
        Gets similarity between user1 and user2, where their interaction vectors with products
        are represented by bitarrays.
        0 indicates no interaction, 1 indicates iteraction.
        """
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

    def num_items_in_common(self, x, y):
        user_x = self._user_bitarrays[x]
        user_y = self._user_bitarrays[y]
        return (user_x & user_y).count()
