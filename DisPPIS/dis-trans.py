import os
import numpy as np
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

pdb_folder = 'pdb'  
output_folders = {
    'distances': 'distances'
}

def ensure_output_folders():
    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)

def read_pdb_sequence(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('PDB', pdb_path)
    sequence = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True):
                    sequence.append(seq1(residue.get_resname(), undef_code='X'))
    return "".join(sequence)

def process_pdb_file(pdb_file_path):
    pdb_id = os.path.splitext(os.path.basename(pdb_file_path))[0]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file_path)

    atoms = [atom for atom in structure.get_atoms() if atom.get_parent().id[0] == ' ']
    coords = np.array([atom.coord for atom in atoms])
    distances = squareform(pdist(coords))

    np.save(os.path.join(output_folders['distances'], f'{pdb_id}.npy'), distances)

if __name__ == "__main__":
    ensure_output_folders()
    pdb_files = [os.path.join(pdb_folder, f) for f in os.listdir(pdb_folder) if f.endswith('.pdb')]

    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        process_pdb_file(pdb_file)
