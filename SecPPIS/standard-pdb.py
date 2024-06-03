import os
import pandas as pd

def extract_cryst1_line(pdb_lines):
    for line in pdb_lines:
        if line.startswith("CRYST1"):
            return line
    return ""

def renumber_atoms(lines):
    new_lines = []
    atom_number = 1
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            new_line = line[:6] + f"{atom_number:5d}" + line[11:]
            new_lines.append(new_line)
            atom_number += 1
        else:
            new_lines.append(line)
    return new_lines

def split_pdb_chains(input_dir, output_dir, csv_file_path):
    df = pd.read_csv(csv_file_path)
    expected_names = df['name'].str[:4].str.upper().unique()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdb"):
            file_base_name = filename[:-4].upper()
            if file_base_name in expected_names:
                filepath = os.path.join(input_dir, filename)
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    cryst1_line = extract_cryst1_line(lines)
                    chains = {}
                    for line in lines:
                        if line.startswith("ATOM") or line.startswith("HETATM"):
                            chain_id = line[21]
                            if chain_id not in chains:
                                chains[chain_id] = [cryst1_line] if cryst1_line else []
                            chains[chain_id].append(line)

                    for chain_id, chain_lines in chains.items():
                        chain_lines = renumber_atoms(chain_lines)
                        chain_file_name = f"{file_base_name}{chain_id}.pdb"
                        chain_file_path = os.path.join(output_dir, chain_file_name)
                        with open(chain_file_path, 'w') as chain_file:
                            chain_file.writelines(chain_lines)
                            print(f"Saved {chain_file_name}")
            else:
                print(f"Skipping {filename} as it is not listed in the CSV file.")

    df['name'] = df['name'].str.upper()
    new_csv_file_path = os.path.join(os.getcwd(), 'new-422.csv')
    df.to_csv(new_csv_file_path, index=False)
    print(f"New CSV file saved as {new_csv_file_path}")

input_dir = 'original-pdb'
output_dir = 'pdb'
csv_file_path = '422_name_seq_label.csv'

split_pdb_chains(input_dir, output_dir, csv_file_path)

