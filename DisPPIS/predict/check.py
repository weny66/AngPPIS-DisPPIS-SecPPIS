import numpy as np
import os
from tqdm import tqdm

def check_data_indices(seq_dir, vocab_size):
    seq_files = sorted(os.listdir(seq_dir))
    out_of_range_indices = {}
    
    for seq_file in tqdm(seq_files, desc='Checking data indices'):
        seq_path = os.path.join(seq_dir, seq_file)
        seq = np.load(seq_path)
        out_of_range = seq[seq >= vocab_size]
        if len(out_of_range) > 0:
            out_of_range_indices[seq_file] = out_of_range.tolist()
    
    return out_of_range_indices

vocab_size = 1000  
seq_dir = "seq1"

out_of_range_indices = check_data_indices(seq_dir, vocab_size)

for file, indices in out_of_range_indices.items():
    print(f"{file} contains out of range indices: {indices}")
