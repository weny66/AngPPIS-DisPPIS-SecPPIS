import os
import pandas as pd

aa_map = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_dir = './prediction_results'
for file in os.listdir(input_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)
        check_columns = []
        if 'A+D+S_results' in df.columns:
            check_columns.append('A+D+S_results')
        if 'A+D_results' in df.columns:
            check_columns.append('A+D_results')
        if not check_columns and 'A+S_results' in df.columns:
            check_columns.append('A+S_results')

        if not check_columns:
            continue  

        results = []
        ranges = {}
        for col in check_columns:
            temp_ranges = []
            for idx, row in df.iterrows():
                if row[col] == 1:
                    start = max(0, idx - 1)
                    end = min(len(df), idx + 2)
                    temp_ranges.append((start, end))

            temp_ranges = sorted(temp_ranges, key=lambda x: x[0])
            merged_ranges = []
            for current in temp_ranges:
                if not merged_ranges or merged_ranges[-1][1] < current[0] - 1:
                    merged_ranges.append(current)
                else:
                    merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], current[1]))
            ranges[col] = merged_ranges

            for start, end in merged_ranges:
                seq = df.loc[start:end, 'AminoAcidName'].str.upper()
                seq = ''.join([aa_map.get(x, 'X') for x in seq])
                results.append([col, seq, f"{start}-{end}", ''])

        output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.txt")
        with open(output_file, 'w') as f:
            f.write("source,motif,range,prediction\n")  
            for result in results:
                f.write(f"{result[0]},{result[1]},{result[2]},{result[3]}\n")  
