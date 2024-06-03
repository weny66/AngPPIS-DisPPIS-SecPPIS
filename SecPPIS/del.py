import os
import pandas as pd

def clean_pdb_files_and_update_csv(pdb_dir, csv_file_path):
    df = pd.read_csv(csv_file_path)
    keep_files = df['name'].str.upper().tolist()  
    all_pdb_files = {file[:-4].upper() for file in os.listdir(pdb_dir) if file.endswith(".pdb")}
    files_to_delete = all_pdb_files - set(keep_files)
    for file_base in files_to_delete:
        for file in os.listdir(pdb_dir):  
            if file.startswith(file_base):
                os.remove(os.path.join(pdb_dir, file))
                print(f"Deleted {file}")
    remaining_pdb_files = {file[:-4].upper() for file in os.listdir(pdb_dir) if file.endswith(".pdb")}

    df = df[df['name'].str.upper().isin(remaining_pdb_files)]
    df.to_csv(csv_file_path, index=False)
    print(f"Updated CSV file saved as {csv_file_path}")
    print("File cleanup and CSV update complete.")

pdb_dir = 'pdb'
csv_file_path = 'new-422.csv'

clean_pdb_files_and_update_csv(pdb_dir, csv_file_path)
