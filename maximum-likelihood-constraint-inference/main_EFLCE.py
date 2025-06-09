import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import random

def preprocess_and_filter_trajectories(csv_file, frame_rate=25, x_threshold=80, y_threshold=-80):
    
    df = pd.read_csv(csv_file)

    # Initialize the result list
    filtered_transitions = []

    # Process data for each unique trackId
    for track_id, group in df.groupby("trackId"):
        # Sort by frame to ensure correct time sequence
        group = group.sort_values("frame")

        # Check if the last state satisfies the filtering condition
        final_state = group.iloc[-1]
        first_state = group.iloc[0]
        if final_state["xCenter"] < x_threshold and final_state["yCenter"] < y_threshold and (first_state["xCenter"] > 160 or first_state["yCenter"] > -40):
            # Extract relevant columns for processing
            states = group[["xCenter", "yCenter", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].values

            # Create transitions (current_state -> next_state)
            for i in range(len(states) - 5):
                current_state = states[i]
                next_state = states[i + 5]
                j = i
                while i - 10 < j < i + 10 or j < 0 or j > len(states) - 1:
                    j = random.randint(i - 20, i + 20)
                random_state = states[j]
                filtered_transitions.append([np.hstack((current_state, next_state)), 1])
                filtered_transitions.append([np.hstack((current_state, random_state)), 0])

    print(f"Filtered {len(filtered_transitions)} transitions from trajectories that end with x < {x_threshold} and y < {y_threshold}.")
    return filtered_transitions

input_csv = "./inD/00_tracks.csv"
filtered_transitions = preprocess_and_filter_trajectories(input_csv)

for i, (input_features, label) in enumerate(filtered_transitions[:5]):
    print(f"Filtered Transition {i + 1}: Current {input_features}, Next {label}")


class TransitionPredictionNN(nn.Module):
    def __init__(self, input_dim=14):
        super(TransitionPredictionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


def split_data(transitions, train_ratio=0.7, val_ratio=0.15):

    # Extract current and next states
    input_states = torch.tensor([t[0] for t in transitions], dtype=torch.float32)
    labels = torch.tensor([t[1] for t in transitions], dtype=torch.float32)

    # Split the data
    train_x, temp_x, train_y, temp_y = train_test_split(input_states, labels, test_size=(1 - train_ratio))
    val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=val_ratio / (1 - train_ratio))

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64, shuffle=False, pin_memory=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=64, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = split_data(filtered_transitions)


def train_and_evaluate(model, optimizer, train_loader, val_loader, epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device).float()

            with torch.cuda.amp.autocast():
                logits = model(batch_features).squeeze(dim=-1)
                loss = criterion(logits, batch_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device).float()
                logits = model(batch_features).squeeze(dim=-1)
                loss = criterion(logits, batch_targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    return train_loss_history, val_loss_history

def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0

    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device).float()
            
            predictions = model(batch_features).squeeze(dim=-1)
            loss = criterion(predictions, batch_targets)
            total_loss += loss.item()

    test_loss = total_loss / len(test_loader)
    print(f"Test Loss = {test_loss:.4f}")
    return test_loss

def plot_errors(train_loss, val_loss, test_loss):
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.legend()
    plt.grid()
    plt.show()


def train_model_allfeatures(epochs = 50, lr = 0.001):
    
    model = TransitionPredictionNN(input_dim=12)
    optimizer = optim.Adam(model.parameters(), lr)   
    # Load and process trajectory data
    train_loader, val_loader, test_loader = split_data(filtered_transitions)
    train_loss, val_loss = train_and_evaluate(model, optimizer, train_loader, val_loader, epochs)

    # Train model
    test_loss = test_model(model, test_loader)


    # Plot loss history
    plot_errors(train_loss, val_loss, test_loss)
    return model

model = train_model_allfeatures(200, lr = 0.0001)