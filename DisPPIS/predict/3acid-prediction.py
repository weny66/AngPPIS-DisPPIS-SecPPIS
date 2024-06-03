import pandas as pd
import os

def process_and_adjust_sequence_numbers(df):
    min_sequence_num = df['AminoAcidSeqNum'].min()
    df['SeqNum'] = df['AminoAcidSeqNum'] - min_sequence_num + 1
    return df

def process_and_save_results(input_csv, output_folder):
    df = pd.read_csv(input_csv)
    df = process_and_adjust_sequence_numbers(df)
    grouped = df.groupby('SeqNum')
    results_list = []

    for name, group in grouped:
        prediction = 1 if any(group['Prediction'] == 1) else 0
        new_row = {
            'SeqNum': name,
            'AminoAcidName': group['AminoAcidName'].iloc[0],
            'Prediction': prediction
        }
        results_list.append(new_row)

    results = pd.DataFrame(results_list)

    base_name = os.path.basename(input_csv).split('_')[0]  
    output_csv = f'{output_folder}/{base_name}_distance_prediction_results.csv'
    results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

def main():
    temp_dir = 'output/temp'
    output_folder = 'output'
    for csv_file in os.listdir(temp_dir):
        if csv_file.endswith('_updated.csv'):
            input_csv = os.path.join(temp_dir, csv_file)
            process_and_save_results(input_csv, output_folder)

if __name__ == "__main__":
    main()
