import numpy as np
import os

def check_dimensions(input_folder):
    file_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.npy')]
    dimensions = []

    for path in file_paths:
        matrix = np.load(path)
        dimensions.append(matrix.shape)
    
    unique_dimensions = set(dimensions)

    if len(unique_dimensions) == 1:
        print(f"same dimension: {unique_dimensions.pop()}")
    else:
        print("different dimension")
        for dim in unique_dimensions:
            print(f"dimension: {dim}, number: {dimensions.count(dim)}")

input_folder = 'input'
check_dimensions(input_folder)
