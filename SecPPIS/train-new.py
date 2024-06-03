import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.metrics import homogeneity_score, completeness_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_samples


def pad_sequences(sequences, maxlen, dtype='float32', padding='post', value=0.):
    num_samples = len(sequences)
    x = np.full((num_samples, maxlen), value, dtype=dtype)
    for i, s in enumerate(sequences):
        if not len(s):
            continue
        trunc_len = min(len(s), maxlen)  
        if padding == 'post':
            x[i, :trunc_len] = s[:trunc_len]
        elif padding == 'pre':
            x[i, -trunc_len:] = s[-trunc_len:]
    return x


def load_data_and_labels(feature_dir, label_dir):
    features, labels = [], []
    max_length = 0
    for file_name in os.listdir(feature_dir):
        feature_path = os.path.join(feature_dir, file_name)
        if os.path.exists(feature_path):
            feature = np.load(feature_path)
            max_length = max(max_length, feature.shape[0])
            features.append(feature)
            if os.path.exists(os.path.join(label_dir, file_name)):
                label = np.load(os.path.join(label_dir, file_name))
                labels.append(label)
    return features, labels, max_length


def pad_features_and_labels(features, labels, max_length):
    padded_features = pad_sequences(features, maxlen=max_length, padding='post')
    padded_labels = pad_sequences(labels, maxlen=max_length, padding='post', dtype='float32')
    return padded_features, padded_labels

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


class BiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        dropped = self.dropout(lstm_out)
        dense_out = torch.relu(self.fc1(dropped))
        final_out = torch.sigmoid(self.fc2(dense_out))
        return final_out


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            optimizer.zero_grad()
            outputs = model(inputs.long())
            loss = criterion(outputs, labels.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
                outputs = model(inputs.long())
                loss = criterion(outputs, labels.unsqueeze(-1))
                val_loss += loss.item()
        
        tqdm.write(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}')


def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.long()).squeeze()  
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.numpy())
    return true_labels, predictions



def calculate_metrics(true_labels_flat, predictions_flat):
    precision = precision_score(true_labels_flat, predictions_binary)
    recall = recall_score(true_labels_flat, predictions_binary)
    f1 = f1_score(true_labels_flat, predictions_binary)
    accuracy = accuracy_score(true_labels_flat, predictions_binary)
    conf_matrix = confusion_matrix(true_labels_flat, predictions_binary)
    print(f'Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nAccuracy: {accuracy}\nConfusion Matrix:\n{conf_matrix}')


def main():
    epochs = 10
    feature_dir, label_dir = 'secondary_structures', 'labels1'
    features, labels, max_length = load_data_and_labels(feature_dir, label_dir)
    padded_features, padded_labels = pad_features_and_labels(features, labels, max_length)
    X_train, X_test, y_train, y_test = train_test_split(padded_features, padded_labels, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = BiLSTM(input_dim=8, embedding_dim=32, hidden_dim=64, output_dim=1)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    metrics_path = os.path.join(output_dir, 'epoch_metrics.csv')
    with open(metrics_path, 'w') as f:
        f.write('Epoch,Precision,Recall,F1,Accuracy,AUROC,AUPRC,MCC\n')

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            optimizer.zero_grad()
            outputs = model(inputs.long())
            loss = criterion(outputs, labels.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Validation", leave=False):
                outputs = model(inputs.long())
                loss = criterion(outputs, labels.unsqueeze(-1))
                val_loss += loss.item()
        
        tqdm.write(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss / len(test_loader)}')

        true_labels, predictions = evaluate_model(model, test_loader)
        true_labels_flat = np.concatenate(true_labels).ravel()
        predictions_flat = np.concatenate(predictions).ravel()

        roc_auc = roc_auc_score(true_labels_flat, predictions_flat)
        auprc = average_precision_score(true_labels_flat, predictions_flat)
        predictions_binary = [1 if x >= 0.5 else 0 for x in predictions_flat]
        precision = precision_score(true_labels_flat, predictions_binary)
        recall = recall_score(true_labels_flat, predictions_binary)
        f1 = f1_score(true_labels_flat, predictions_binary)
        accuracy = accuracy_score(true_labels_flat, predictions_binary)
        mcc = matthews_corrcoef(true_labels_flat, predictions_binary)

        with open(metrics_path, 'a') as f:
            f.write(f'{epoch+1},{precision},{recall},{f1},{accuracy},{roc_auc},{auprc},{mcc}\n')

        model_save_path = os.path.join(output_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    main()

