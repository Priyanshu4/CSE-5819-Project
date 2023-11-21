from similarity import *
import numpy as np

class AnomalyScore:
    def __init__(self,users, mapping):
        self.leaves = users
        self.mapping = mapping

    def penalty_function(self):
        pass

    def review_tightness(self):
        pass

    def product_tightness(self):
        pass

    def neighbor_tightness(self):
        pass
    