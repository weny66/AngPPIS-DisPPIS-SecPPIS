import numpy as np
import os

def split_and_save_elements(label_file_path, output_folder):
    labels = np.load(label_file_path)

    filename_without_extension = os.path.splitext(os.path.basename(label_file_path))[0]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for label_index, label in enumerate(labels):
        label_filename = f"{filename_without_extension}_{label_index}.npy"
        label_path = os.path.join(output_folder, label_filename)
        np.save(label_path, np.array([label]))

if __name__ == "__main__":
    input_folder = 'labels'  
    output_folder = 'lab'  

    for label_file in os.listdir(input_folder):
        if label_file.endswith('.npy'):
            label_file_path = os.path.join(input_folder, label_file)
            split_and_save_elements(label_file_path, output_folder)
    print("finish")
