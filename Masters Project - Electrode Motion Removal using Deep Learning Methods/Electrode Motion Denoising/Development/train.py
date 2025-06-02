# Simplified and optimized ECG denoising training script

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import RCNN
import numpy as np
import h5py
from data_loading import normalise, load_h5_file, SimpleECGDataset
import os 

# Device setup
device = 'mps'
print(f'Using device: {device}')

# Define paths (update these)
clean_signals_path = '/Users/benrussell/Library/CloudStorage/GoogleDrive-rben3625@gmail.com/My Drive/cleanSignals/'
noisy_signals_path = '/Users/benrussell/Library/CloudStorage/GoogleDrive-rben3625@gmail.com/My Drive/noisySignals/'

# Custom dataset
class SimpleECGDataset(torch.utils.data.Dataset):
    def __init__(self, clean_files, noisy_files, segment_length=1500):
        self.clean_files = clean_signals_path
        self.noisy_signals_files = noisy_signals_path
        self.segment_length = segment_length

        self.clean_files = sorted(os.listdir(clean_signals_path))
        self.noisy_files = sorted(os.listdir(noisy_signals_path))

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_signal = load_h5_file(os.path.join(clean_signals_path, self.clean_files[idx]))
        noisy_signal = load_h5_file(os.path.join(noisy_signals_path, self.noisy_files[idx]))

        if clean_signal is None or noisy_signal is None:
            raise ValueError(f"Error loading signal pair at index {idx}")

        clean_signal = normalise(clean_signal[:self.segment_length])
        noisy_signal = normalise(noisy_signal[:self.segment_length])

        return torch.tensor(noisy_signal, dtype=torch.float32), torch.tensor(clean_signal, dtype=torch.float32)

# Dataset and loader
dataset = SimpleECGDataset(clean_signals_path, noisy_signals_path)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Load model
from models import RCNN
model = RCNN().to(device)

# Loss and optimizer
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.002)
criterion = torch.nn.MSELoss()

# Custom QRS-weighted loss (optional)
def qrs_loss(y_pred, y_true, peaks, a=20):
    loss = criterion(y_pred, y_true)
    for peak in peaks:
        peak_range = slice(max(0, peak-35), min(peak+37, len(y_true)))
        loss += 20 * criterion(y_pred[peak_range], y_true[peak_range])
    return loss

# Training loop
def train(model, dataloader, epochs=20):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            output = model(noisy.unsqueeze(1).float())
            loss = criterion(output, clean)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.6f}')

# Run training
train(model, dataloader)
