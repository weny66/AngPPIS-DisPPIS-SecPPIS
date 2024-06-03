import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd

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
        lstm_out, _ = self.lstm(embedded)
        dropped = self.dropout(lstm_out)
        dense_out = torch.relu(self.fc1(dropped))
        final_out = torch.sigmoid(self.fc2(dense_out))
        return final_out

input_dim = 8  
embedding_dim = 32
hidden_dim = 64
output_dim = 1
model_load_path = 'model/model_epoch_10.pth'


model = BiLSTM(input_dim, embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(model_load_path))
model.eval()  

input_folder = './input'

for file_name in os.listdir(input_folder):
    if file_name.endswith('.npy'):
        file_path = os.path.join(input_folder, file_name)
        features = np.load(file_path)
        features_tensor = torch.tensor(features, dtype=torch.long).unsqueeze(0)  
        predictions = []  
        with torch.no_grad():
            output = model(features_tensor)
            predicted_labels = (output.squeeze() > 0.5).int().numpy()  
        positions = range(1, len(predicted_labels) + 1)  
        predictions.extend(zip(positions, predicted_labels))

        base_file_name = file_name.rsplit('.', 1)[0]  
        output_file = f'output/{base_file_name}_secondary-structure_prediction_results.csv'
        df = pd.DataFrame(predictions, columns=['Position', 'Prediction'])
        df.to_csv(output_file, index=False)
        print(f'Predictions saved to {output_file}')
