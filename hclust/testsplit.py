import numpy as np

# Create a large NumPy array (your data)
large_matrix = np.array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9, 10],
                         [11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20]])

# Calculate the number of rows
num_rows = large_matrix.shape[0]

# Shuffle the row indices
shuffled_indices = np.random.permutation(num_rows)

# Calculate the number of rows for each group
group_size = num_rows // 3

# Split the shuffled indices into three equal-sized groups
group1_indices = shuffled_indices[:group_size]
group2_indices = shuffled_indices[group_size:2 * group_size]
group3_indices = shuffled_indices[2 * group_size:]

# Use the indices to split the matrix into three groups
group1 = large_matrix[group1_indices]
group2 = large_matrix[group2_indices]
group3 = large_matrix[group3_indices]

# Print the three groups
print("Group 1:")
print(group1)

print("\nGroup 2:")
print(group2)

print("\nGroup 3:")
print(group3)
