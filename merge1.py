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
    if len(group_files) < 3:
        print(f"jump {group_name}，only {len(group_files)} ")
        continue
    columns_data = {
        'SeqNum': [],
        'AminoAcidName': [],
        'angles_prediction_results': [],
        'distance_prediction_results': [],
        'secondary-structure_prediction_results': []
    }

    for file in group_files:
        df = pd.read_csv(file)
        if 'distance' in file:
            columns_data['SeqNum'].extend(df['SeqNum'])
            columns_data['AminoAcidName'].extend(df['AminoAcidName'])
            columns_data['distance_prediction_results'].extend(df['Prediction'])
        elif 'angles' in file:
            columns_data['angles_prediction_results'].extend(df['Prediction'])
        elif 'secondary' in file:
            columns_data['secondary-structure_prediction_results'].extend(df['Prediction'])

    max_length = max(len(v) for v in columns_data.values())
    for key, value in columns_data.items():
        columns_data[key] += [None] * (max_length - len(value))

    merged_df = pd.DataFrame(columns_data)

    merged_df['A+D+S_results'] = merged_df.apply(lambda row: 1 if (row['angles_prediction_results'] == 1 
                                                                    and row['distance_prediction_results'] == 1 
                                                                    and row['secondary-structure_prediction_results'] == 1) 
                                                  else 0, axis=1)
    merged_df['A+D_results'] = merged_df.apply(lambda row: 1 if (row['angles_prediction_results'] == 1 
                                                                  and row['distance_prediction_results'] == 1) 
                                               else 0, axis=1)

    results_path = os.path.join(output_dir, f'{group_name}_results.csv')
    merged_df.to_csv(results_path, index=False)
    print(f'save {results_path}')
