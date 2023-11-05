import numpy as np
import pickle

def unpickle(file_name):
    try:
        with open(file_name, "rb") as file:
            loaded_object = pickle.load(file)
            print("Successfully loaded the pickled object:")
            return loaded_object
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred while loading the pickled object: {e}")

# Example usage:
file = "../data/yelpnyc/embedded/deepfd/embs_ep10.pkl"
obj = unpickle(file)
print(len(obj))
print(type(obj))