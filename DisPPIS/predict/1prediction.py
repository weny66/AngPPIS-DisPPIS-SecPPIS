import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import re


def load_data(seq_dir, lab_dir=None, max_length=None):
    sequences = []
    labels = []
    filenames = []  
    seq_files = sorted(os.listdir(seq_dir))
    for seq_file in tqdm(seq_files, desc='Loading data'):
        seq_path = os.path.join(seq_dir, seq_file)
        seq = np.load(seq_path)
        sequences.append(seq)
        filenames.append(seq_file)  
        if lab_dir is not None:  
            lab_path = os.path.join(lab_dir, seq_file.replace('.npy', '.npy'))
            label = np.load(lab_path)
            labels.append(label[0])
    
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    sequences_padded = np.zeros((len(sequences), max_length))
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        sequences_padded[i, :length] = seq[:length]

    if lab_dir is not None:
        return sequences_padded, np.array(labels), max_length
    else:
        return sequences_padded, None, max_length, filenames

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim, max_length):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 32, 5)
        self.pool = nn.MaxPool1d(4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * ((max_length - 5 + 1) // 4), 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def extract_features(self, x):
        """Extract features from the penultimate layer"""
        x = self.embedding(x).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return x

def load_model(model_path, vocab_size, embed_dim, output_dim, max_length, device):
    model = CNNModel(vocab_size, embed_dim, output_dim, max_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, dataloader, device):
    predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs.to(device))
            predicted_probs = torch.sigmoid(outputs)  
            predictions.extend(predicted_probs.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'model/model_epoch_14.pth'  
    vocab_size = 1000  
    embed_dim = 32  
    output_dim = 1  
    max_length = 13512  

    with open('vocab_size.txt', 'r') as f:
        vocab_size = int(f.read())

    model = load_model(model_path, vocab_size, embed_dim, output_dim, max_length, device)

    seq_dir = 'seq'  
    sequences, _, max_length, filenames = load_data(seq_dir, None, max_length=max_length)
    sequences_tensor = torch.tensor(sequences, dtype=torch.long).to(device)
    predict_loader = DataLoader(sequences_tensor, batch_size=32, shuffle=False)

    predictions = predict(model, predict_loader, device)
    pred_labels = (predictions > 0.6).astype(int)

    temp_dir = 'output/temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    def extract_full_number(s):
        numbers = re.findall(r'\d+', s)
        return int(''.join(numbers)) if numbers else 0

    groups = {}
    for filename, prediction in zip(filenames, pred_labels.flatten()):
        group_name = filename[:4]
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append((filename, prediction))

    temp_dir = 'output/temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for group_name, items in groups.items():
        sorted_items = sorted(items, key=lambda x: extract_full_number(x[0]))
        sorted_filenames, sorted_predictions = zip(*sorted_items)
        df = pd.DataFrame({'Filename': sorted_filenames, 'Prediction': sorted_predictions})
        csv_filename = f'{temp_dir}/{group_name}_prediction_results.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Predictions for group {group_name} saved to {csv_filename}")

if __name__ == "__main__":
    main()