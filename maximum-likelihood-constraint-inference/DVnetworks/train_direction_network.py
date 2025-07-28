import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# ==== 1. Preprocessing and Training ====
def prepare_data(df, features, target, cat_col):
    le = LabelEncoder()
    df['vehicle_class_idx'] = le.fit_transform(df[cat_col])

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    y_cos = np.cos(df[target].values).astype(np.float32)
    y_sin = np.sin(df[target].values).astype(np.float32)
    y = np.stack([y_cos, y_sin], axis=1)

    X_cat = df['vehicle_class_idx'].values
    X_num = df[features].values.astype(np.float32)

    X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
        X_cat, X_num, y, test_size=0.2, random_state=42
    )

    train_ds = TensorDataset(
        torch.tensor(X_cat_train, dtype=torch.long),
        torch.tensor(X_num_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_cat_val, dtype=torch.long),
        torch.tensor(X_num_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    return train_loader, val_loader, le, scaler

class AnglePredictor(nn.Module):
    def __init__(self, num_classes, embed_dim=4):
        super().__init__()
        self.vehicle_embed = nn.Embedding(num_classes, embed_dim)
        self.fc1 = nn.Linear(embed_dim + 5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, vehicle_class_idx, numeric_inputs):
        embed = self.vehicle_embed(vehicle_class_idx)
        x = torch.cat([embed, numeric_inputs], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def cosine_sine_loss(pred, target):
    return ((pred - target) ** 2).sum(dim=1).mean()

def train_model(model, train_loader, val_loader, device, model_path="angle_model.pt", epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_cat, x_num, y in train_loader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x_cat, x_num)
            loss = cosine_sine_loss(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_cat)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_cat, x_num, y in val_loader:
                x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
                output = model(x_cat, x_num)
                loss = cosine_sine_loss(output, y)
                val_loss += loss.item() * len(x_cat)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# ==== 2. Plotting Predictions ====
def plot_predictions(df, model, le, scaler, features, device, plot_path="angle_predictions.png", N=100):
    sample_df = df.sample(N, random_state=42).reset_index(drop=True)
    sample_df['vehicle_class_idx'] = le.transform(sample_df['vehicle_class'])

    X_cat_sample = torch.tensor(sample_df['vehicle_class_idx'].values, dtype=torch.long).to(device)
    X_num_sample = torch.tensor(scaler.transform(sample_df[features]), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred_cos_sin = model(X_cat_sample, X_num_sample).cpu().tolist()
        pred_cos_sin = np.array(pred_cos_sin)

    pred_angles = np.arctan2(pred_cos_sin[:, 1], pred_cos_sin[:, 0])
    true_angles = sample_df['future_angle_rad'].values

    x = sample_df['current_x'].values
    y = sample_df['current_y'].values

    true_dx = np.cos(true_angles)
    true_dy = np.sin(true_angles)
    pred_dx = np.cos(pred_angles)
    pred_dy = np.sin(pred_angles)

    plt.figure(figsize=(10, 10))
    plt.quiver(x, y, true_dx, true_dy, color='blue', angles='xy', scale_units='xy', scale=1, width=0.003, label='True')
    plt.quiver(x, y, pred_dx, pred_dy, color='red', angles='xy', scale_units='xy', scale=1, width=0.003, label='Predicted')
    plt.title('Predicted vs True Future Angles')
    plt.xlabel('current_x')
    plt.ylabel('current_y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

df = pd.read_csv('direction_training_data_detailed.csv')
features = ['current_x', 'current_y', 'current_vx', 'current_vy', 'current_angle_rad']
target = 'future_angle_rad'
train_loader, val_loader, le, scaler = prepare_data(df, features, target, 'vehicle_class')
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnglePredictor(num_classes=len(le.classes_)).to(device)
train_model(model, train_loader, val_loader, device, model_path="angle_model.pt", epochs=10)
plot_predictions(df, model, le, scaler, features, device, plot_path="angle_predictions.png", N=100) 