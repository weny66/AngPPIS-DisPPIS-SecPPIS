import numpy as np
import os
import random

def balance_labels(label_folder):
    labels_count = {0: [], 1: []}
    for label_file in os.listdir(label_folder):
        if label_file.endswith('.npy'):
            label = np.load(os.path.join(label_folder, label_file))
            labels_count[label[0]].append(label_file)

    count_0 = len(labels_count[0])
    count_1 = len(labels_count[1])

    print(f"number - 0: {count_0}, 1: {count_1}")

    excess_category = 0 if count_0 > count_1 else 1
    files_to_remove = random.sample(labels_count[excess_category], abs(count_0 - count_1))

    for file in files_to_remove:
        os.remove(os.path.join(label_folder, file))

    print(f"After balance - 0: {min(count_0, count_1)}, 1: {min(count_0, count_1)}")

def match_and_clean_folders(folder_a, folder_b):
    files_a = set([os.path.splitext(f)[0] for f in os.listdir(folder_a)])
    files_b = set([os.path.splitext(f)[0] for f in os.listdir(folder_b)])

    for file in files_a - files_b:
        os.remove(os.path.join(folder_a, file + '.npy'))

    for file in files_b - files_a:
        os.remove(os.path.join(folder_b, file + '.npy'))

if __name__ == "__main__":
    label_folder = 'lab'  
    seq_folder = 'seq'  

    balance_labels(label_folder)
    match_and_clean_folders(label_folder, seq_folder)
    print("finish")
