import pandas as pd
import numpy as np
import json

df = pd.read_csv("../data/yelpnyc/yelpnyc/metadata.csv")

average_ratings = df.groupby('Product_id2')['Rating'].mean().to_dict()

print(average_ratings)
average_ratings_json = json.dumps(average_ratings)

file_path = 'average_ratings.json'  # Replace with your desired file path
with open(file_path, 'w') as file:
    file.write(average_ratings_json)

print(f"JSON data written to '{file_path}'")