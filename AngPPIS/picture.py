import cv2
import numpy as np
import os
from tqdm import tqdm

def convert_matrices_to_images_cubic(input_dir, output_dir, target_size=(256, 256)):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filenames = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    for filename in tqdm(filenames, desc="Processing"):
        file_path = os.path.join(input_dir, filename)
        matrix = np.load(file_path)

        min_val, max_val = np.min(matrix), np.max(matrix)
        matrix_normalized = (matrix - min_val) / (max_val - min_val) * 255  
        image = np.uint8(matrix_normalized)

        image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
        
        output_filename = os.path.splitext(filename)[0] + '.png'
        cv2.imwrite(os.path.join(output_dir, output_filename), image_resized)

input_dir = 'input'  
output_dir = 'images_cubic'  
convert_matrices_to_images_cubic(input_dir, output_dir, target_size=(256, 256))
