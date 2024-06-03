import os
import numpy as np

def count_labels_and_ratios(folder_path):
    file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.npy')]
    
    total_ones = 0
    total_zeros = 0
    file_count = len(file_paths)
    
    for file_path in file_paths:
        labels = np.load(file_path)
        ones_count = np.sum(labels == 1)
        zeros_count = np.sum(labels == 0)
        total_ones += ones_count
        total_zeros += zeros_count
        ratio = ones_count / zeros_count if zeros_count > 0 else "Inf"
        print(f"{os.path.basename(file_path)}: 1 = {ones_count}, 0 = {zeros_count}, ratio（1:0）= {ratio}")

    avg_ratio = (total_ones / total_zeros) if total_zeros > 0 else "Inf"
    print(f"average ratio（1:0）= {avg_ratio}")
    print(f"total 1 = {total_ones}, 0 = {total_zeros}")

labels_folder_path = 'labels'
count_labels_and_ratios(labels_folder_path)
