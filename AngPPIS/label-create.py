import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
import os
from Bio.SeqUtils import seq1  

def read_pdb_sequence(pdb_path):
    parser = PDBParser()
    structure = parser.get_structure('PDB', pdb_path)
    sequence = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() not in ['HOH', 'WAT']:  
                    sequence.append(seq1(residue.get_resname()))
    return "".join(sequence)

def align_labels(pdb_dir, csv_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    labels_df = pd.read_csv(csv_path)
    for index, row in labels_df.iterrows():
        pdb_name = row['name']
        label_str = row['label']
        pdb_sequence = row['sequence']
        pdb_path = os.path.join(pdb_dir, f"{pdb_name}.pdb")
        
        if os.path.exists(pdb_path):
            sequence = read_pdb_sequence(pdb_path)
            start = sequence.find(pdb_sequence)
            if start != -1:
                end = start + len(pdb_sequence)
                if len(label_str) < len(pdb_sequence):
                    label_str = label_str.ljust(len(pdb_sequence), '0')
                elif len(label_str) > len(pdb_sequence):
                    label_str = label_str[:len(pdb_sequence)]
                label_str = '0' * start + label_str + '0' * (len(sequence) - end)
            else:
                label_str = '0' * len(sequence)

            np.save(os.path.join(output_dir, f"{pdb_name}.npy"), np.array(list(label_str), dtype=int))

pdb_dir = 'pdb'  
csv_path = 'new-422.csv'  
output_dir = 'labels'  
align_labels(pdb_dir, csv_path, output_dir)
