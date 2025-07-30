#!/usr/bin/env python3
"""
Clean module containing only the constraint model functions needed for RL.
This avoids importing the training code from main_DVnetworks.py.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import random
import hashlib
import json
from scipy.ndimage import gaussian_filter1d

class TransitionPredictionNN(nn.Module):
    """
    Neural network for predicting transition likelihood (constraint model).
    This must match the exact architecture used in main_DVnetworks.py.
    """
    def __init__(self, input_dim=23):  # 14 base features + 9 collision features
        super(TransitionPredictionNN, self).__init__()
        
        # Architecture: input_dim -> 128 -> 64 -> 32 -> 1 (matching main_DVnetworks.py)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.fc(x)

def load_model(filepath):
    """
    Load a trained model from file
    
    Args:
        filepath: Path to the saved model file
    
    Returns:
        model: Loaded PyTorch model
        metadata: Model metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    try:
        model_data = torch.load(filepath, map_location='cpu')
        
        # Create model with correct architecture
        model = TransitionPredictionNN(input_dim=model_data['model_config']['input_dim'])
        model.load_state_dict(model_data['model_state_dict'])
        
        print(f"Loaded model from {filepath}")
        print(f"  Parameters: {model_data['training_params']}")
        print(f"  Created: {model_data['created_at']}")
        
        return model, model_data
        
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        raise

def get_nearby_vehicles(df, ego_track_id, frame, max_neighbors=5):
    """
    Get nearby vehicles for collision avoidance features.
    
    Args:
        df: DataFrame with vehicle data
        ego_track_id: ID of ego vehicle
        frame: Current frame number
        max_neighbors: Maximum number of nearby vehicles to consider
    
    Returns:
        nearby_vehicles: List of nearby vehicle data
    """
    # Get all vehicles in the current frame (excluding ego)
    frame_data = df[df['frame'] == frame]
    other_vehicles = frame_data[frame_data['trackId'] != ego_track_id]
    
    if len(other_vehicles) == 0:
        return []
    
    # Get ego vehicle position (if available)
    ego_data = frame_data[frame_data['trackId'] == ego_track_id]
    if len(ego_data) == 0:
        # If ego not in frame, use first vehicle as reference
        ego_pos = [other_vehicles.iloc[0]['xCenter'], other_vehicles.iloc[0]['yCenter']]
    else:
        ego_pos = [ego_data.iloc[0]['xCenter'], ego_data.iloc[0]['yCenter']]
    
    # Calculate distances to ego
    distances = []
    for _, vehicle in other_vehicles.iterrows():
        vehicle_pos = [vehicle['xCenter'], vehicle['yCenter']]
        distance = np.linalg.norm(np.array(vehicle_pos) - np.array(ego_pos))
        distances.append((distance, vehicle))
    
    # Sort by distance and take closest neighbors
    distances.sort(key=lambda x: x[0])
    nearby_vehicles = []
    
    for i, (distance, vehicle) in enumerate(distances[:max_neighbors]):
        nearby_vehicles.append({
            'x': vehicle['xCenter'],
            'y': vehicle['yCenter'],
            'vx': vehicle['xVelocity'],
            'vy': vehicle['yVelocity'],
            'ax': vehicle['xAcceleration'],
            'ay': vehicle['yAcceleration'],
            'distance': distance,
            'track_id': vehicle['trackId']
        })
    
    return nearby_vehicles

def create_collision_avoidance_features(current_state, next_state, other_vehicles, current_frame, next_frame):
    """
    Create collision avoidance features for constraint model input.
    
    Args:
        current_state: Current vehicle state [x, y, vx, vy, ax, ay, ...]
        next_state: Next vehicle state [x, y, vx, vy, ax, ay, ...]
        other_vehicles: List of nearby vehicle data
        current_frame: Current frame number
        next_frame: Next frame number
    
    Returns:
        collision_features: List of 9 collision avoidance features
    """
    if not other_vehicles:
        # No other vehicles, return zeros
        return [0.0] * 9
    
    # Extract ego vehicle states
    ego_current_pos = np.array([current_state[0], current_state[1]])  # x, y
    ego_next_pos = np.array([next_state[0], next_state[1]])
    ego_current_vel = np.array([current_state[2], current_state[3]])  # vx, vy
    ego_next_vel = np.array([next_state[2], next_state[3]])
    
    # Calculate collision features for each nearby vehicle
    collision_features = []
    
    for vehicle in other_vehicles:
        # Vehicle current and predicted positions
        vehicle_current_pos = np.array([vehicle['x'], vehicle['y']])
        vehicle_current_vel = np.array([vehicle['vx'], vehicle['vy']])
        
        # Predict vehicle position at next frame (simple linear prediction)
        dt = (next_frame - current_frame) * 0.04  # Assuming 25fps
        vehicle_next_pos = vehicle_current_pos + vehicle_current_vel * dt
        
        # Distance features
        current_distance = np.linalg.norm(ego_current_pos - vehicle_current_pos)
        next_distance = np.linalg.norm(ego_next_pos - vehicle_next_pos)
        
        # Relative velocity
        relative_vel = ego_next_vel - vehicle_current_vel
        relative_speed = np.linalg.norm(relative_vel)
        
        # Time to collision (simplified)
        if relative_speed > 0.1:  # Avoid division by zero
            ttc = next_distance / relative_speed
        else:
            ttc = float('inf')
        
        # Add features for this vehicle
        collision_features.extend([
            current_distance,
            next_distance,
            relative_speed,
            ttc if ttc != float('inf') else 100.0,  # Cap at 100
            vehicle['distance'],  # Original distance
            vehicle['vx'],
            vehicle['vy'],
            vehicle['ax'],
            vehicle['ay']
        ])
    
    # Pad with zeros if we have fewer than max_neighbors vehicles
    while len(collision_features) < 9:
        collision_features.append(0.0)
    
    # Take only the first 9 features (for max_neighbors=1)
    return collision_features[:9] 