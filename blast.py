import os
import pandas as pd

aa_mapping = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def convert_and_save_sequences(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)
            if 'AminoAcidName' in df.columns:
                df = df.dropna(subset=['AminoAcidName'])
                df['AminoAcid'] = df['AminoAcidName'].map(aa_mapping)
                sequence = ''.join(df['AminoAcid'].astype(str))  
                output_filename = os.path.splitext(filename)[0] + '.txt'
                output_file_path = os.path.join(output_folder, output_filename)
                with open(output_file_path, 'w') as f:
                    f.write(sequence)
                print(f"Sequence saved to {output_file_path}")

input_folder_path = 'prediction_results'
output_folder_path = 'output-seq'

convert_and_save_sequences(input_folder_path, output_folder_path)
