import pickle

# Load (unpickle) data from the "data.pkl" file
with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# View the loaded data
print(loaded_data)
