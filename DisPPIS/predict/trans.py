import os
import numpy as np
from Bio.PDB import PDBParser, is_aa
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm  

pdb_folder = 'pdb'  
output_folder = 'distances'  

def ensure_output_folder(folder):
    os.makedirs(folder, exist_ok=True)

def calculate_residue_atom_counts(structure):
    atom_counts = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True):
                    atom_counts.append(len(residue))
    return atom_counts

def process_pdb_file(pdb_file_path):
    pdb_id = os.path.splitext(os.path.basename(pdb_file_path))[0]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file_path)

    atoms = [atom for atom in structure.get_atoms() if atom.get_parent().id[0] == ' ']
    coords = np.array([atom.coord for atom in atoms])
    distances = squareform(pdist(coords))
    
    np.save(os.path.join(output_folder, f'{pdb_id}.npy'), distances)

if __name__ == "__main__":
    ensure_output_folder(output_folder)
    pdb_files = [os.path.join(pdb_folder, f) for f in os.listdir(pdb_folder) if f.endswith('.pdb')]
    
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        process_pdb_file(pdb_file)
