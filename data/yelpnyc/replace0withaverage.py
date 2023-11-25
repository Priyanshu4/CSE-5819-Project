import pickle
from scipy.sparse import csr_matrix
import numpy as np


with open("data/yelpnyc/yelpnyc_ratings.pkl", "rb") as file:
    ratings_matrix = pickle.load(file)

with open("data/yelpnyc/yelpnyc_avg_ratings.pkl", "rb") as file:
    average_ratings = pickle.load(file)

ratings_matrix_avg = ratings_matrix.toarray().copy()
for row in ratings_matrix_avg:
    mask = row == 0
    row[mask] = average_ratings[mask]

print(ratings_matrix_avg)

with open("yelpnyc_ratings.pkl", "wb") as file:
    pickle.dump(ratings_matrix_avg, file, protocol=4)
