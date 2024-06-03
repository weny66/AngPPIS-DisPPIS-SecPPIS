import os
import numpy as np
from Bio.PDB import PDBParser, DSSP
from tqdm import tqdm

def create_feature_directory(feature_name):
    os.makedirs(feature_name, exist_ok=True)

def extract_and_save_secondary_structures(filename, pdb_id):
    parser = PDBParser()
    structure = parser.get_structure('', filename)
    model = structure[0]
    dssp = DSSP(model, filename, dssp='/usr/bin/mkdssp')

    secondary_structures = []
    for chain in model:
        for res in chain:
            if res.id[0] == ' ':
                dssp_key = (chain.id, res.id[1])
                sec_struc = secondary_structure_mapping.get(dssp[dssp_key][2], 0) if dssp_key in dssp else 0
                secondary_structures.append(sec_struc)

    np.save(os.path.join('secondary_structures1', f'{pdb_id}.npy'), secondary_structures)

def process_pdb_files(pdb_folder):
    feature_name = 'secondary_structures1'
    create_feature_directory(feature_name)
    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith('.pdb')]
    
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        pdb_id = os.path.splitext(pdb_file)[0]
        filename = os.path.join(pdb_folder, pdb_file)
        extract_and_save_secondary_structures(filename, pdb_id)

secondary_structure_mapping = {
    'H': 1,  
    'B': 2, 
    'E': 3,  
    'G': 4,  
    'I': 5,  
    'T': 6, 
    'S': 7, 
    '-': 0   
}

if __name__ == "__main__":
    pdb_folder = 'input'  
    process_pdb_files(pdb_folder)

