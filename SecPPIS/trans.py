import os
import numpy as np
from Bio.PDB import PDBParser, DSSP
from tqdm import tqdm

def three_to_one(three_letter_code):
    conversion = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    }
    return conversion.get(three_letter_code.upper(), 'X')

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

def extract_and_save_features(filename, pdb_id):
    parser = PDBParser()
    try:
        structure = parser.get_structure('', filename)
    except Exception as e:
        print(f"Error parsing PDB file {filename}: {e}")
        return  

    try:
        model = structure[0]
        dssp = DSSP(model, filename, dssp='/usr/bin/mkdssp')  

        secondary_structures = []
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':
                    dssp_key = (chain.id, res.id[1])
                    if dssp_key in dssp:
                        sec_struc = secondary_structure_mapping.get(dssp[dssp_key][2], 0)
                    else:
                        sec_struc = 0
                    secondary_structures.append(sec_struc)
        np.save(os.path.join('secondary_structures', f'{pdb_id}.npy'), secondary_structures)

    except Exception as e:
        print(f"Failed to process {filename} with DSSP: {e}")


def process_pdb_files(pdb_folder):
    feature_names = ['secondary_structures']
    for feature_name in feature_names:
        os.makedirs(feature_name, exist_ok=True)
    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith('.pdb')]
    
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        pdb_id = os.path.splitext(pdb_file)[0]
        filename = os.path.join(pdb_folder, pdb_file)
        extract_and_save_features(filename, pdb_id)

if __name__ == "__main__":
    pdb_folder = 'pdb'  
    process_pdb_files(pdb_folder)

