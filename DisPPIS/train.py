import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import homogeneity_score, completeness_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm
import os

BATCH_SIZE = 10
LEARNING_RATE = 0.00005
NUM_EPOCHS = 15
EMBED_DIM = 32
NUM_CLASSES = 1
RANDOM_STATE = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(seq_dir, lab_dir):
    sequences = []
    labels = []
    max_length = 0
    seq_files = sorted(os.listdir(seq_dir))
    for seq_file in tqdm(seq_files, desc='Loading data'):
        seq_path = os.path.join(seq_dir, seq_file)
        lab_path = os.path.join(lab_dir, seq_file.replace('.npy', '.npy'))
        seq = np.load(seq_path)
        label = np.load(lab_path)
        sequences.append(seq)
        labels.append(label[0])
        if len(seq) > max_length:
            max_length = len(seq)
    sequences_padded = np.zeros((len(sequences), max_length))
    for i, seq in enumerate(sequences):
        sequences_padded[i, :len(seq)] = seq
    return sequences_padded, np.array(labels), max_length

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

def save_epoch_metrics(epoch_metrics, output_file):
    df = pd.DataFrame(epoch_metrics)
    df.to_csv(output_file, index=False)

def calculate_clustering_metrics(features, labels, epoch, output_dir):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar()
    plt.title(f"t-SNE visualization of features for Epoch {epoch}")
    plt.savefig(f"{output_dir}/tsne_epoch_{epoch}.png")
    plt.close()

    kmeans = KMeans(n_clusters=2, random_state=42).fit(features)  
    labels_pred = kmeans.labels_

    homogeneity = homogeneity_score(labels, labels_pred)
    completeness = completeness_score(labels, labels_pred)
    silhouette = silhouette_score(features, labels_pred)  
    calinski_harabaz = calinski_harabasz_score(features, labels_pred)
    davies_bouldin = davies_bouldin_score(features, labels_pred)

    clustering_metrics = {
        "Homogeneity": homogeneity,
        "Completeness": completeness,
        "Silhouette Coefficient": silhouette,
        "Calinski-Harabasz Index": calinski_harabaz,
        "Davies-Bouldin Index": davies_bouldin,
    }

    clustering_file_path = f"{output_dir}/clustering_metrics.txt"
    file_exists = os.path.exists(clustering_file_path)
    with open(clustering_file_path, 'a') as f:
        if not file_exists:
            f.write("Epoch,Homogeneity,Completeness,Silhouette Coefficient,Calinski-Harabasz Index,Davies-Bouldin Index\n")
        f.write(f"{epoch},{clustering_metrics['Homogeneity']},{clustering_metrics['Completeness']},{clustering_metrics['Silhouette Coefficient']},{clustering_metrics['Calinski-Harabasz Index']},{clustering_metrics['Davies-Bouldin Index']}\n")


    return clustering_metrics


def main():
    seq_dir = 'seq'
    lab_dir = 'lab'
    sequences, labels, max_length = load_data(seq_dir, lab_dir)

    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=RANDOM_STATE)
    X_train_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  

    vocab_size = int(np.max(sequences) + 1)
    model = CNNModel(vocab_size, EMBED_DIM, NUM_CLASSES, max_length).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    output_metrics_file = 'output/metrics.csv'
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    epoch_metrics = []

    if not os.path.exists(output_metrics_file):
        with open(output_metrics_file, 'w') as f:
            f.write("Epoch,Accuracy,Precision,Recall,F1,AUC-ROC,AUPRC,MCC\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0  
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Training"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)  
        print(f"Epoch {epoch+1}: Train Loss = {average_train_loss:.4f}")

        torch.save(model.state_dict(), f"{output_dir}/model_epoch_{epoch+1}.pth")
        features, labels = get_features_and_labels(model, test_loader)
        clustering_metrics = calculate_clustering_metrics(features, labels, epoch+1, output_dir)

        evaluation_metrics = evaluate_model(model, X_test_tensor, y_test_tensor, BATCH_SIZE)

        epoch_metrics.append({
            "Epoch": epoch + 1,
            **evaluation_metrics,  
            **clustering_metrics  
        })

        with open(output_metrics_file, 'a') as f:
            f.write(f"{epoch+1},{evaluation_metrics['Accuracy']},{evaluation_metrics['Precision']},{evaluation_metrics['Recall']},{evaluation_metrics['F1']},{evaluation_metrics['AUROC']},{evaluation_metrics['AUPRC']},{evaluation_metrics['MCC']}\n")
    vocab_size = int(np.max(sequences) + 1)
    with open('vocab_size.txt', 'w') as f:
        f.write(str(vocab_size))


def evaluate_model(model, X_test, y_test, batch_size):
    model.eval()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    labels_list = []
    total_test_loss = 0  
    criterion = nn.BCEWithLogitsLoss()  

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)  
            total_test_loss += loss.item()  
            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    average_test_loss = total_test_loss / len(test_loader)  

    
    pred_labels = (np.array(predictions) > 0.5).astype(int)
    roc_auc = roc_auc_score(labels_list, predictions)
    auprc = average_precision_score(labels_list, predictions)
    mcc = matthews_corrcoef(labels_list, pred_labels)
    accuracy = accuracy_score(labels_list, pred_labels)
    precision = precision_score(labels_list, pred_labels)
    recall = recall_score(labels_list, pred_labels)
    f1 = f1_score(labels_list, pred_labels)

    print(f"Average Test Loss: {average_test_loss:.4f}")
    return {
        "AUROC": roc_auc,
        "AUPRC": auprc,
        "MCC": mcc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

def get_features_and_labels(model, dataloader):
    model.eval()  
    features = []
    labels = []
    with torch.no_grad():  
        for inputs, labels_batch in dataloader:
            inputs = inputs.to(device)
            labels.extend(labels_batch.cpu().numpy())  
            feature_batch = model.extract_features(inputs)  
            features.extend(feature_batch.cpu().numpy())  
            
    return np.array(features), np.array(labels)


if __name__ == "__main__":
    main()
