import pandas as pd
import os
from glob import glob

def extract_amino_acid_sequence(pdb_file):
    amino_acid_sequence = []
    observed_amino_acids = set()
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                residue_id = line[22:26].strip()
                amino_acid = line[17:20].strip()
                if residue_id not in observed_amino_acids:
                    observed_amino_acids.add(residue_id)
                    amino_acid_sequence.append(amino_acid)
    return amino_acid_sequence

def add_amino_acid_sequence_and_update_filename(csv_file, pdb_folder, output_dir):
    base_name = os.path.basename(csv_file)[:4].lower()
    pdb_files = glob(os.path.join(pdb_folder, f'{base_name}*.pdb'))
    pdb_files = [pdb_file for pdb_file in pdb_files if pdb_file.split(os.sep)[-1][:4].lower() == base_name]

    if pdb_files:
        amino_acid_sequence = extract_amino_acid_sequence(pdb_files[0])
        df = pd.read_csv(csv_file)

        df['NumSeq'] = df['Filename'].apply(lambda x: int(x.split('_')[-1].split('.')[0]))
        df.drop('Filename', axis=1, inplace=True)

        amino_acid_len = len(amino_acid_sequence)
        df['AminoAcidName'] = [amino_acid_sequence[i] if i < amino_acid_len else None for i in range(len(df))]

        df = df[['NumSeq', 'AminoAcidName', 'Prediction']]

        new_file_path = os.path.join(output_dir, os.path.basename(csv_file))
        df.to_csv(new_file_path, index=False)
        print(f'finish {new_file_path}')
    else:
        print(f'no-match：{base_name}')

input_csv_dir = 'output/temp'  
pdb_folder = 'pdb'  
output_dir = 'output'  

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for csv_file in glob(os.path.join(input_csv_dir, '*.csv')):
    add_amino_acid_sequence_and_update_filename(csv_file, pdb_folder, output_dir)
