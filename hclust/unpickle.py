import pickle, os

os.chdir("/Users/niteeshsaravanan/Documents/GitHub/CSE-5819-Project/hclust/results/2023-11-03_14-08-38")
#os.chdir("/Users/niteeshsaravanan/Documents/GitHub/CSE-5819-Project/hclust/results/2023-11-03_02-45-55")
# Load (unpickle) data from the "data.pkl" file
try:
    with open('linkage.pkl', 'rb') as file:
        linkage = pickle.load(file)
    print(linkage)
except:
    pass

with open('leaves.pkl', 'rb') as file:
    leaves = pickle.load(file)


# View the loaded data

print(leaves)
