
import numpy as np
import os

def split_and_save_columns(matrix_file_path, output_folder):
    matrix = np.load(matrix_file_path)
    filename_without_extension = os.path.splitext(os.path.basename(matrix_file_path))[0]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    for col_index in range(matrix.shape[1]):
        column_filename = f"{filename_without_extension}_{col_index}.npy"
        column_path = os.path.join(output_folder, column_filename)
        np.save(column_path, matrix[:, col_index])

if __name__ == "__main__":
    input_folder = 'distances'  
    output_folder = 'seq'  
    for matrix_file in os.listdir(input_folder):
        if matrix_file.endswith('.npy'):
            matrix_file_path = os.path.join(input_folder, matrix_file)
            split_and_save_columns(matrix_file_path, output_folder)
    print("finish")
