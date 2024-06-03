import pandas as pd
from Bio.PDB import PDBParser
import os

def read_csv_file(csv_filename):
    return pd.read_csv(csv_filename)

def get_amino_acids_from_pdb(pdb_filename):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("my_structure", pdb_filename)
    amino_acids_info = {}
    for model in structure:
        for chain in model:
            for residue in chain.get_residues():
                if residue.id[0] == " ":
                    residue_id = residue.get_id()[1]
                    residue_name = residue.get_resname()
                    for atom in residue:
                        atom_serial_num = atom.get_serial_number()
                        amino_acids_info[atom_serial_num] = (residue_id, residue_name)
    return amino_acids_info

def match_amino_acids_to_csv(df, amino_acids_info):
    df['AminoAcidNum'] = df['Filename'].apply(lambda x: int(x.split('_')[-1].split('.')[0]))
    df['AminoAcidInfo'] = df['AminoAcidNum'].apply(lambda x: amino_acids_info.get(x+1, (None, '')))
    df['AminoAcidSeqNum'] = df['AminoAcidInfo'].apply(lambda x: x[0])
    df['AminoAcidName'] = df['AminoAcidInfo'].apply(lambda x: x[1])
    df.drop('AminoAcidInfo', axis=1, inplace=True)
    return df

def main():
    temp_dir = 'output/temp'
    pdb_dir = 'pdb'
    
    pdb_files = {os.path.splitext(f)[0]: os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb')}
    
    for csv_file in os.listdir(temp_dir):
        if csv_file.endswith('.csv'):
            csv_filename = os.path.join(temp_dir, csv_file)
            group_name = csv_file.split('_')[0]  
            pdb_filename = pdb_files.get(group_name)
            
            if pdb_filename:
                df = read_csv_file(csv_filename)
                amino_acids_info = get_amino_acids_from_pdb(pdb_filename)
                df_matched = match_amino_acids_to_csv(df, amino_acids_info)
                
                updated_csv_filename = os.path.join(temp_dir, f"{group_name}_updated.csv")
                df_matched.to_csv(updated_csv_filename, index=False)
                print(f"Updated CSV saved to {updated_csv_filename}")

if __name__ == "__main__":
    main()
