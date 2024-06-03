import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import homogeneity_score, completeness_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import average_precision_score, matthews_corrcoef
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.labels = [f.replace('.png', '.npy') for f in self.img_names]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        image = Image.open(img_path).convert('RGB')
        label = np.load(label_path)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label.squeeze()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_dir = 'images_cubic'
label_dir = 'lab'
full_dataset = CustomImageDataset(img_dir, label_dir, transform=transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 2) 
)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
metrics_file = os.path.join(output_dir, 'metrics.txt')
with open(metrics_file, 'w') as f:
    f.write("Epoch,Accuracy,Precision,Recall,F1 Score,AUC-ROC,MCC\n")

def evaluate_and_visualize(model, dataloader, device, epoch, output_dir):
    model.eval()
    y_true = []
    y_pred = []
    y_probs_list = []
    features = []
    total_loss = 0 
    criterion = torch.nn.CrossEntropyLoss()  


    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)  
            total_loss += loss.item()  
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs_list.extend(outputs.cpu().numpy())
            features.extend(outputs.cpu().numpy())

    average_loss = total_loss / len(dataloader)  
    print(f"Test Loss: {average_loss:.4f}")  

    features = StandardScaler().fit_transform(np.array(features))
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_true, cmap='viridis', alpha=0.6)
    plt.colorbar()
    plt.title(f"t-SNE visualization of features at Epoch {epoch}")
    plt.savefig(os.path.join(output_dir, f"tsne_epoch_{epoch}.png"))
    plt.close()


    homogeneity = homogeneity_score(y_true, y_pred)
    completeness = completeness_score(y_true, y_pred)
    silhouette = silhouette_score(features, y_pred)
    calinski_harabaz = calinski_harabasz_score(features, y_pred)
    davies_bouldin = davies_bouldin_score(features, y_pred)
    clustering_metrics_file = os.path.join(output_dir, 'clustering_metrics.txt')
    with open(clustering_metrics_file, 'a') as cmf:
        cmf.write(f"{epoch},{homogeneity:.4f},{completeness:.4f},{silhouette:.4f},{calinski_harabaz:.4f},{davies_bouldin:.4f}\n")


    y_probs = [prob[1] for prob in y_probs_list]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc_roc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nAUC-ROC: {auc_roc}")

    return accuracy, precision, recall, f1, auc_roc, auprc, mcc


for epoch in range(10):  
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    accuracy, precision, recall, f1, auc_roc, auprc, mcc = evaluate_and_visualize(model, test_loader, device, epoch+1, output_dir)

    with open(metrics_file, 'a') as f:
        f.write(f"{epoch+1},{accuracy:.4f},{precision:.4f},{recall:.4f},{f1:.4f},{auc_roc:.4f},{auprc:.4f},{mcc:.4f}\n")
    torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))



