import numpy as np
import pickle

def unpickle_and_print(file_name):
    try:
        with open(file_name, "rb") as file:
            loaded_object = pickle.load(file)
            print("Successfully loaded the pickled object:")
            print(loaded_object)
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred while loading the pickled object: {e}")

# Example usage:
file_name = "hclust/results/2023-11-05_00-57-23/leaves.pkl"
unpickle_and_print(file_name)
