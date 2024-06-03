import pandas as pd
import os
from glob import glob

output_dir = 'prediction_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_dir = 'input/'
files = glob(os.path.join(input_dir, '*.csv'))

groups = {}
for file in files:
    group_name = os.path.basename(file)[:4]
    if group_name not in groups:
        groups[group_name] = []
    groups[group_name].append(file)

for group_name, group_files in groups.items():
    if len(group_files) != 2:
        print(f"save {group_name}，only {len(group_files)} ")
        continue
    columns_data = {
        'NumSeq': [],
        'AminoAcidName': [],
        'angles_prediction': [],
        'secondary_prediction': []
    }
    angles_processed = False
    secondary_processed = False

    for file in group_files:
        if 'angles' in file and not angles_processed:
            df = pd.read_csv(file)
            columns_data['NumSeq'] = df['NumSeq']
            columns_data['AminoAcidName'] = df['AminoAcidName']
            columns_data['angles_prediction'] = df['Prediction']
            angles_processed = True
        elif 'secondary' in file and not secondary_processed:
            df = pd.read_csv(file)
            columns_data['secondary_prediction'] = df['Prediction']
            secondary_processed = True
    
    if not angles_processed or not secondary_processed:
        print(f" {group_name} loss necessary")
        continue

    merged_df = pd.DataFrame(columns_data)
    merged_df['merge-results'] = merged_df.apply(lambda row: 1 if row['angles_prediction'] == 1 and row['secondary_prediction'] == 1 else 0, axis=1)

    results_path = os.path.join(output_dir, f'{group_name}_results.csv')
    merged_df.to_csv(results_path, index=False)
    print(f'save {results_path}')
