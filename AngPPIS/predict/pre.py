import torch
from torchvision import transforms, models
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch.nn as nn
import numpy as np
import os
from PIL import Image

class CustomPredictDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [img_name for img_name in os.listdir(img_dir) if img_name.endswith('.png')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.img_names[idx]

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 2)  
)


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    filenames = []
    with torch.no_grad():
        for inputs, img_names in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            filenames.extend(img_names)
    return filenames, predictions

def group_filenames(filenames, predictions):
    groups = {}
    for filename, prediction in zip(filenames, predictions):
        group_name = filename[:4]  
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append((filename, prediction))
    return groups

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'model/model_epoch_4.pth'
    
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)  
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = CustomPredictDataset(img_dir='images_cubic', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    filenames, predictions = predict(model, dataloader, device)

    groups = group_filenames(filenames, predictions)
    for group_name, group_items in groups.items():
        sorted_group_items = sorted(group_items, key=lambda x: int(x[0].split('_')[-1].split('.')[0]))
        sorted_filenames, sorted_predictions = zip(*sorted_group_items)
        df = pd.DataFrame({'Filename': sorted_filenames, 'Prediction': sorted_predictions})
        csv_filename = f'output/{group_name}_angles_prediction_results.csv'
        df.to_csv(csv_filename, index=False)

if __name__ == "__main__":
    main()
