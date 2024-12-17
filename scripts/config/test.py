import yaml
import numpy as np

# Load the YAML file
with open("test.yaml", "r") as file:
    data = yaml.safe_load(file)

# Access the matrix
matrix1 = data.get("matrix1")
# Print the matrix
print("Loaded Matrix1:")
for row in matrix1:
    print(row)
numpy_matrix1 = np.array(matrix1)
print("numpy_matrix1: ", numpy_matrix1)


matrix2 = data.get("matrix2")
# Print the matrix
print("Loaded Matrix2:")
for row in matrix2:
    print(row)
numpy_matrix2 = np.array(matrix2)
print("numpy_matrix2: ", numpy_matrix2)
