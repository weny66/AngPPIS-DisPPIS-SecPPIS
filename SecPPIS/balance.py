import os
import numpy as np

def calculate_dynamic_expand_range(zeros, ones):
    diff = zeros - ones
    if diff <= 0:
        return 0
    expand_range = diff // ones
    return expand_range

def dynamic_oversampling(labels, expand_range):
    if expand_range <= 0:
        return labels
    expanded_labels = np.copy(labels)
    for i in range(len(labels)):
        if labels[i] == 1:
            start = max(i - expand_range, 0)
            end = min(i + expand_range + 1, len(labels))
            expanded_labels[start:end] = 1
    return expanded_labels

def process_labels_folder_dynamic(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.npy'):
            file_path = os.path.join(input_folder, file_name)
            labels = np.load(file_path)
            
            zeros = np.sum(labels == 0)
            ones = np.sum(labels == 1)
            expand_range = calculate_dynamic_expand_range(zeros, ones)

            expanded_labels = dynamic_oversampling(labels, expand_range)

            output_file_path = os.path.join(output_folder, file_name)
            np.save(output_file_path, expanded_labels)
            print(f"Processed {file_name}: expand_range = {expand_range}, saved to {output_file_path}")
input_folder = 'labels'
output_folder = 'labels1'
process_labels_folder_dynamic(input_folder, output_folder)
