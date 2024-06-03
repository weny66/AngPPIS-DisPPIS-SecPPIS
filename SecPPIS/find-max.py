import numpy as np
import os

def find_max_length(seq_dir):
    max_length = 0
    for seq_file in os.listdir(seq_dir):
        if seq_file.endswith('.npy'):
            seq_path = os.path.join(seq_dir, seq_file)
            seq = np.load(seq_path)
            if len(seq) > max_length:
                max_length = len(seq)
    return max_length

if __name__ == "__main__":
    seq_dir = 'secondary_structures'  
    max_length = find_max_length(seq_dir)
    print(f"The maximum length of sequences in '{seq_dir}' is: {max_length}")
