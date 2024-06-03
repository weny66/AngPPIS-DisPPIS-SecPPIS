import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, is_aa
import os
from Bio.SeqUtils import seq1  

def read_pdb_sequence(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('PDB', pdb_path)
    sequence = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() not in ['HOH', 'WAT'] and is_aa(residue, standard=True):  
                    sequence.append(seq1(residue.get_resname()))
    return "".join(sequence)

def calculate_residue_atom_counts(structure):
    atom_counts = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True):
                    atom_counts.append(len(residue))
    return atom_counts

def replicate_labels_to_atoms(labels, atom_counts):
    replicated_labels = ''.join([label * count for label, count in zip(labels, atom_counts)])
    return replicated_labels

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
            structure = PDBParser(QUIET=True).get_structure(pdb_name, pdb_path)
            sequence = read_pdb_sequence(pdb_path)
            atom_counts = calculate_residue_atom_counts(structure)
            start = sequence.find(pdb_sequence)
            if start != -1:
                end = start + len(pdb_sequence)
                adjusted_label_str = '0' * start + label_str + '0' * (len(sequence) - end)
            else:
                adjusted_label_str = '0' * len(sequence)  

            if len(adjusted_label_str) < len(sequence):
                adjusted_label_str = adjusted_label_str.ljust(len(sequence), '0')
            elif len(adjusted_label_str) > len(sequence):
                adjusted_label_str = adjusted_label_str[:len(sequence)]

            replicated_labels = replicate_labels_to_atoms(adjusted_label_str, atom_counts)  

            np.save(os.path.join(output_dir, f"{pdb_name}.npy"), np.array([int(label) for label in replicated_labels], dtype=int))

pdb_dir = 'pdb'  
csv_path = 'new-422.csv'  
output_dir = 'labels'  
align_labels(pdb_dir, csv_path, output_dir)
