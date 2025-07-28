#!/usr/bin/env python3
"""
Specialized Neural Networks for Trajectory Prediction
1. Direction Network: Predicts velocity direction (angle) at t+40
2. Amplitude Network: Predicts velocity magnitude at t+40

Both networks use 40-frame smoothing and predict 40 frames into the future.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
SMOOTHING_FRAMES = 40  # Number of frames for smoothing
PREDICTION_HORIZON = 40  # Frames to predict into the future
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100

class DirectionPredictionNN(nn.Module):
    """
    Neural network to predict velocity direction (angle) at t+40
    Input: [x, y, velocity_angle] (current position and velocity direction)
    Output: velocity_angle at t+40
    """
    def __init__(self, input_dim=3):
        super(DirectionPredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class AmplitudePredictionNN(nn.Module):
    """
    Neural network to predict velocity magnitude at t+40
    Input: [x, y, velocity_magnitude, closest_car_distance, moving_towards_ego]
    Output: velocity_magnitude at t+40
    """
    def __init__(self, input_dim=5):
        super(AmplitudePredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def calculate_velocity_angle(vx, vy):
    """Calculate velocity direction angle in radians"""
    return torch.atan2(vy, vx)

def calculate_velocity_magnitude(vx, vy):
    """Calculate velocity magnitude"""
    return torch.sqrt(vx**2 + vy**2)

def smooth_trajectory(trajectory, window_size=SMOOTHING_FRAMES):
    """
    Apply moving average smoothing to trajectory data
    """
    if len(trajectory) < window_size:
        return trajectory
    
    smoothed = []
    for i in range(len(trajectory)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(trajectory), i + window_size // 2 + 1)
        window = trajectory[start_idx:end_idx]
        smoothed.append(np.mean(window, axis=0))
    
    return np.array(smoothed)

def save_smoothed_trajectories(track_df, tracks_meta_df=None, output_file='smoothed_trajectories.csv'):
    """
    Save smoothed trajectory data for inspection
    """
    print("Saving smoothed trajectory data...")
    
    smoothed_data = []
    
    for track_id, group in track_df.groupby('trackId'):
        trajectory = group.values
        
        # Get vehicle class from metadata if available
        vehicle_class = "unknown"
        if tracks_meta_df is not None:
            track_meta = tracks_meta_df[tracks_meta_df['trackId'] == track_id]
            if len(track_meta) > 0:
                vehicle_class = track_meta.iloc[0]['class']
        
        # Smooth the trajectory
        smoothed_traj = smooth_trajectory(trajectory)
        
        # Calculate velocity angles and magnitudes
        vx = smoothed_traj[:, 9]  # xVelocity
        vy = smoothed_traj[:, 10]  # yVelocity
        velocity_angles = np.arctan2(vy, vx)
        velocity_magnitudes = np.sqrt(vx**2 + vy**2)
        
        for i in range(len(smoothed_traj)):
            row = {
                'track_id': track_id,
                'vehicle_class': vehicle_class,
                'frame': smoothed_traj[i, 2],
                'xCenter': smoothed_traj[i, 4],
                'yCenter': smoothed_traj[i, 5],
                'xVelocity': vx[i],
                'yVelocity': vy[i],
                'velocity_angle_rad': velocity_angles[i],
                'velocity_angle_deg': np.degrees(velocity_angles[i]),
                'velocity_magnitude': velocity_magnitudes[i],
                'original_xCenter': trajectory[i, 4] if i < len(trajectory) else smoothed_traj[i, 4],
                'original_yCenter': trajectory[i, 5] if i < len(trajectory) else smoothed_traj[i, 5],
                'original_xVelocity': trajectory[i, 9] if i < len(trajectory) else smoothed_traj[i, 9],
                'original_yVelocity': trajectory[i, 10] if i < len(trajectory) else smoothed_traj[i, 10]
            }
            smoothed_data.append(row)
    
    smoothed_df = pd.DataFrame(smoothed_data)
    smoothed_df.to_csv(output_file, index=False)
    print(f"Saved smoothed trajectory data to '{output_file}'")
    
    return smoothed_df

def find_closest_car(current_pos, current_frame, all_trajectories, current_track_id):
    """
    Find the closest car to the current position at the same frame
    """
    min_distance = float('inf')
    closest_car_info = None
    
    for track_id, trajectory in all_trajectories.items():
        if track_id == current_track_id:
            continue
            
        # Find position at current frame
        frame_data = trajectory[trajectory[:, 2] == current_frame]  # frame is at index 2
        if len(frame_data) == 0:
            continue
            
        other_pos = frame_data[0, 4:6]  # xCenter, yCenter at indices 4, 5
        distance = np.linalg.norm(current_pos - other_pos)
        
        if distance < min_distance:
            min_distance = distance
            closest_car_info = {
                'distance': distance,
                'position': other_pos,
                'velocity': frame_data[0, 9:11] if len(frame_data[0]) > 11 else np.array([0, 0])  # xVelocity, yVelocity
            }
    
    return closest_car_info

def calculate_relative_motion(current_pos, current_vel, closest_car_pos, closest_car_vel):
    """
    Calculate if closest car is moving towards or away from ego car
    Returns: 1 if moving towards, -1 if moving away, 0 if stationary
    """
    if closest_car_pos is None:
        return 0
    
    # Vector from ego to closest car
    relative_pos = closest_car_pos - current_pos
    relative_vel = closest_car_vel - current_vel
    
    # Dot product to determine direction
    dot_product = np.dot(relative_pos, relative_vel)
    
    if abs(dot_product) < 0.1:  # Threshold for stationary
        return 0
    elif dot_product > 0:
        return 1  # Moving towards
    else:
        return -1  # Moving away

def preprocess_data_for_direction_network(track_df, all_trajectories, tracks_meta_df=None):
    """
    Prepare data for direction prediction network
    """
    print("Preparing data for direction prediction network...")
    
    direction_data = []
    direction_data_detailed = []  # For saving detailed information
    
    for track_id, group in track_df.groupby('trackId'):
        trajectory = group.values
        
        # Get vehicle class from metadata if available
        vehicle_class = "unknown"
        if tracks_meta_df is not None:
            track_meta = tracks_meta_df[tracks_meta_df['trackId'] == track_id]
            if len(track_meta) > 0:
                vehicle_class = track_meta.iloc[0]['class']
        
        # Smooth the trajectory
        smoothed_traj = smooth_trajectory(trajectory)
        
        # Calculate velocity angles from xVelocity and yVelocity
        vx = smoothed_traj[:, 9]  # xVelocity (column 9)
        vy = smoothed_traj[:, 10]  # yVelocity (column 10)
        velocity_angles = np.arctan2(vy, vx)
        
        # Create training pairs
        for i in range(len(smoothed_traj) - PREDICTION_HORIZON):
            current_frame = smoothed_traj[i, 2]  # frame (column 2)
            future_frame = smoothed_traj[i + PREDICTION_HORIZON, 2]
            
            # Input features: [x, y, velocity_angle]
            current_x = smoothed_traj[i, 4]  # xCenter (column 4)
            current_y = smoothed_traj[i, 5]  # yCenter (column 5)
            current_angle = velocity_angles[i]
            
            # Target: future velocity angle
            future_angle = velocity_angles[i + PREDICTION_HORIZON]
            
            # Normalize angles to [-π, π]
            current_angle = np.arctan2(np.sin(current_angle), np.cos(current_angle))
            future_angle = np.arctan2(np.sin(future_angle), np.cos(future_angle))
            
            direction_data.append({
                'input': [current_x, current_y, current_angle],
                'target': future_angle,
                'track_id': track_id,
                'current_frame': current_frame,
                'future_frame': future_frame
            })
            
            # Detailed data for inspection
            direction_data_detailed.append({
                'track_id': track_id,
                'vehicle_class': vehicle_class,
                'current_frame': current_frame,
                'future_frame': future_frame,
                'current_x': current_x,
                'current_y': current_y,
                'current_vx': vx[i],
                'current_vy': vy[i],
                'current_angle_rad': current_angle,
                'current_angle_deg': np.degrees(current_angle),
                'future_angle_rad': future_angle,
                'future_angle_deg': np.degrees(future_angle),
                'angle_change_rad': future_angle - current_angle,
                'angle_change_deg': np.degrees(future_angle - current_angle),
                'prediction_horizon': PREDICTION_HORIZON
            })
    
    print(f"Created {len(direction_data)} direction prediction samples")
    
    # Save detailed direction data
    direction_df = pd.DataFrame(direction_data_detailed)
    direction_df.to_csv('direction_training_data_detailed.csv', index=False)
    print(f"Saved detailed direction training data to 'direction_training_data_detailed.csv'")
    
    return direction_data

def preprocess_data_for_amplitude_network(track_df, all_trajectories, tracks_meta_df=None):
    """
    Prepare data for amplitude prediction network
    """
    print("Preparing data for amplitude prediction network...")
    
    amplitude_data = []
    amplitude_data_detailed = []  # For saving detailed information
    
    for track_id, group in track_df.groupby('trackId'):
        trajectory = group.values
        
        # Get vehicle class from metadata if available
        vehicle_class = "unknown"
        if tracks_meta_df is not None:
            track_meta = tracks_meta_df[tracks_meta_df['trackId'] == track_id]
            if len(track_meta) > 0:
                vehicle_class = track_meta.iloc[0]['class']
        
        # Smooth the trajectory
        smoothed_traj = smooth_trajectory(trajectory)
        
        # Calculate velocity magnitudes from xVelocity and yVelocity
        vx = smoothed_traj[:, 9]  # xVelocity (column 9)
        vy = smoothed_traj[:, 10]  # yVelocity (column 10)
        velocity_magnitudes = np.sqrt(vx**2 + vy**2)
        
        # Create training pairs
        for i in range(len(smoothed_traj) - PREDICTION_HORIZON):
            current_frame = smoothed_traj[i, 2]  # frame (column 2)
            future_frame = smoothed_traj[i + PREDICTION_HORIZON, 2]
            
            # Current state
            current_pos = smoothed_traj[i, 4:6]  # xCenter, yCenter (columns 4, 5)
            current_vel = smoothed_traj[i, 9:11]  # xVelocity, yVelocity (columns 9, 10)
            current_magnitude = velocity_magnitudes[i]
            
            # Find closest car
            closest_car_info = find_closest_car(
                current_pos, current_frame, all_trajectories, track_id
            )
            
            if closest_car_info is None:
                closest_distance = 1000.0  # Default large distance
                moving_towards = 0
            else:
                closest_distance = closest_car_info['distance']
                moving_towards = calculate_relative_motion(
                    current_pos, current_vel,
                    closest_car_info['position'], closest_car_info['velocity']
                )
            
            # Target: future velocity magnitude
            future_magnitude = velocity_magnitudes[i + PREDICTION_HORIZON]
            
            # Input features: [x, y, velocity_magnitude, closest_car_distance, moving_towards_ego]
            amplitude_data.append({
                'input': [current_pos[0], current_pos[1], current_magnitude, closest_distance, moving_towards],
                'target': future_magnitude,
                'track_id': track_id,
                'current_frame': current_frame,
                'future_frame': future_frame
            })
            
            # Detailed data for inspection
            amplitude_data_detailed.append({
                'track_id': track_id,
                'vehicle_class': vehicle_class,
                'current_frame': current_frame,
                'future_frame': future_frame,
                'current_x': current_pos[0],
                'current_y': current_pos[1],
                'current_vx': current_vel[0],
                'current_vy': current_vel[1],
                'current_magnitude': current_magnitude,
                'future_magnitude': future_magnitude,
                'magnitude_change': future_magnitude - current_magnitude,
                'closest_car_distance': closest_distance,
                'moving_towards_ego': moving_towards,
                'prediction_horizon': PREDICTION_HORIZON
            })
    
    print(f"Created {len(amplitude_data)} amplitude prediction samples")
    
    # Save detailed amplitude data
    amplitude_df = pd.DataFrame(amplitude_data_detailed)
    amplitude_df.to_csv('amplitude_training_data_detailed.csv', index=False)
    print(f"Saved detailed amplitude training data to 'amplitude_training_data_detailed.csv'")
    
    return amplitude_data

def train_network(model, train_data, val_data, model_name, epochs=EPOCHS):
    """
    Train a neural network
    """
    print(f"\nTraining {model_name}...")
    
    # Prepare data
    train_inputs = torch.tensor([d['input'] for d in train_data], dtype=torch.float32).to(device)
    train_targets = torch.tensor([d['target'] for d in train_data], dtype=torch.float32).to(device)
    val_inputs = torch.tensor([d['input'] for d in val_data], dtype=torch.float32).to(device)
    val_targets = torch.tensor([d['target'] for d in val_data], dtype=torch.float32).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(train_inputs).squeeze()
        loss = criterion(outputs, train_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_inputs).squeeze()
            val_loss = criterion(val_outputs, val_targets)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
    
    print(f"Training completed for {model_name}")
    return train_losses, val_losses

def save_model(model, model_name, train_losses, val_losses):
    """
    Save trained model and training history
    """
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.fc1.in_features,
            'smoothing_frames': SMOOTHING_FRAMES,
            'prediction_horizon': PREDICTION_HORIZON
        },
        'train_losses': train_losses,
        'val_losses': val_losses
    }, f'{model_name}.pth')
    
    print(f"Model saved as {model_name}.pth")

def load_model(model_class, model_name):
    """
    Load a trained model
    """
    checkpoint = torch.load(f'{model_name}.pth', map_location=device)
    model = model_class(input_dim=checkpoint['model_config']['input_dim']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

def generate_trajectory_with_directional_networks(direction_model, amplitude_model, initial_state, 
                                                track_df, all_trajectories, max_steps=200):
    """
    Generate trajectory using both direction and amplitude networks
    """
    print("Generating trajectory with directional networks...")
    
    trajectory = []
    current_state = initial_state.copy()
    
    for step in range(max_steps):
        # Current position and velocity
        x, y = current_state[0], current_state[1]
        vx, vy = current_state[2], current_state[3]
        
        # Calculate current velocity angle and magnitude
        current_angle = math.atan2(vy, vx)
        current_magnitude = math.sqrt(vx**2 + vy**2)
        
        # Find closest car for amplitude prediction
        closest_car_info = find_closest_car(
            np.array([x, y]), step, all_trajectories, -1  # -1 for generated trajectory
        )
        
        if closest_car_info is None:
            closest_distance = 1000.0
            moving_towards = 0
        else:
            closest_distance = closest_car_info['distance']
            moving_towards = calculate_relative_motion(
                np.array([x, y]), np.array([vx, vy]),
                closest_car_info['position'], closest_car_info['velocity']
            )
        
        # Predict future direction and amplitude
        with torch.no_grad():
            # Direction prediction
            direction_input = torch.tensor([[x, y, current_angle]], dtype=torch.float32).to(device)
            future_angle = direction_model(direction_input).item()
            
            # Amplitude prediction
            amplitude_input = torch.tensor([[x, y, current_magnitude, closest_distance, moving_towards]], 
                                         dtype=torch.float32).to(device)
            future_magnitude = amplitude_model(amplitude_input).item()
        
        # Convert back to velocity components
        future_vx = future_magnitude * math.cos(future_angle)
        future_vy = future_magnitude * math.sin(future_angle)
        
        # Update state (simple Euler integration)
        dt = 0.04  # Assuming 25 FPS
        x_new = x + vx * dt
        y_new = y + vy * dt
        
        # Update current state
        current_state = [x_new, y_new, future_vx, future_vy]
        
        # Add to trajectory
        trajectory.append({
            'frame': step,
            'x': x_new,
            'y': y_new,
            'vx': future_vx,
            'vy': future_vy,
            'angle': future_angle,
            'magnitude': future_magnitude
        })
    
    print(f"Generated trajectory with {len(trajectory)} points")
    return trajectory

def verify_column_structure(track_df):
    """
    Verify that the trajectory data has the expected column structure
    """
    expected_columns = [
        'recordingId', 'trackId', 'frame', 'trackLifetime', 'xCenter', 'yCenter',
        'heading', 'width', 'length', 'xVelocity', 'yVelocity', 'xAcceleration',
        'yAcceleration', 'lonVelocity', 'latVelocity', 'lonAcceleration', 'latAcceleration'
    ]
    
    actual_columns = list(track_df.columns)
    
    print("=== Column Structure Verification ===")
    print(f"Expected columns: {expected_columns}")
    print(f"Actual columns: {actual_columns}")
    
    if actual_columns == expected_columns:
        print("✅ Column structure matches expected format")
        return True
    else:
        print("❌ Column structure mismatch!")
        print("Missing columns:", set(expected_columns) - set(actual_columns))
        print("Extra columns:", set(actual_columns) - set(expected_columns))
        return False

def load_and_filter_tracks(tracks_file='./../inD/00_tracks.csv', tracks_meta_file='./../inD/00_tracksMeta.csv'):
    """
    Load tracks data and filter for cars and truck_buses based on metadata
    """
    print("=== Loading and Filtering Tracks Data ===")
    
    # Load tracks data
    try:
        track_df = pd.read_csv(tracks_file)
        print(f"Loaded {len(track_df)} trajectory points from {tracks_file}")
    except FileNotFoundError:
        print(f"❌ {tracks_file} not found. Please ensure the file exists.")
        return None
    
    # Load tracks metadata
    try:
        tracks_meta_df = pd.read_csv(tracks_meta_file)
        print(f"Loaded {len(tracks_meta_df)} track metadata entries from {tracks_meta_file}")
    except FileNotFoundError:
        print(f"❌ {tracks_meta_file} not found. Using all tracks without filtering.")
        return track_df
    
    # Verify metadata column structure
    print(f"Tracks metadata columns: {list(tracks_meta_df.columns)}")
    
    # Check if 'class' column exists
    if 'class' not in tracks_meta_df.columns:
        print("❌ 'class' column not found in tracks metadata. Using all tracks without filtering.")
        return track_df
    
    # Show available vehicle classes
    available_classes = tracks_meta_df['class'].unique()
    print(f"Available vehicle classes: {available_classes}")
    
    # Filter for cars and truck_buses
    valid_classes = ['car', 'truck_bus']
    filtered_meta = tracks_meta_df[tracks_meta_df['class'].isin(valid_classes)]
    
    print(f"Filtered metadata: {len(filtered_meta)} tracks out of {len(tracks_meta_df)} total")
    print(f"Class distribution in filtered data:")
    print(filtered_meta['class'].value_counts())
    
    # Get track IDs for valid vehicles
    valid_track_ids = filtered_meta['trackId'].unique()
    print(f"Valid track IDs: {len(valid_track_ids)} tracks")
    
    # Filter tracks data
    filtered_tracks = track_df[track_df['trackId'].isin(valid_track_ids)]
    
    print(f"Filtered tracks data: {len(filtered_tracks)} trajectory points out of {len(track_df)} total")
    print(f"Remaining tracks: {filtered_tracks['trackId'].nunique()} unique track IDs")
    
    # Show track length distribution
    track_lengths = filtered_tracks.groupby('trackId').size()
    print(f"Track length statistics:")
    print(f"  Min length: {track_lengths.min()} frames")
    print(f"  Max length: {track_lengths.max()} frames")
    print(f"  Mean length: {track_lengths.mean():.1f} frames")
    print(f"  Median length: {track_lengths.median():.1f} frames")
    
    return filtered_tracks

def main():
    """
    Main function to train and test the directional networks
    """
    print("=== Directional Velocity Networks ===")
    print(f"Smoothing frames: {SMOOTHING_FRAMES}")
    print(f"Prediction horizon: {PREDICTION_HORIZON}")
    
    # Load and filter trajectory data
    track_df = load_and_filter_tracks()
    
    if track_df is None:
        print("❌ Failed to load trajectory data. Please check your files.")
        return
    
    # Verify column structure
    if not verify_column_structure(track_df):
        print("❌ Please check your data format and try again.")
        return
    
    # Load tracks metadata for vehicle class information
    tracks_meta_df = None
    try:
        tracks_meta_df = pd.read_csv('./../inD/00_tracksMeta.csv')
        print(f"Loaded tracks metadata for vehicle class information")
    except FileNotFoundError:
        print("⚠️  Tracks metadata not found, vehicle class information will not be included")
    
    # Create trajectory dictionary for closest car calculations
    all_trajectories = {}
    for track_id, group in track_df.groupby('trackId'):
        all_trajectories[track_id] = group.values
    
    # Save smoothed trajectories for inspection
    smoothed_df = save_smoothed_trajectories(track_df, tracks_meta_df)
    
    # Prepare data for both networks
    
    direction_data = preprocess_data_for_direction_network(track_df, all_trajectories, tracks_meta_df)
    """
    amplitude_data = preprocess_data_for_amplitude_network(track_df, all_trajectories, tracks_meta_df)
    
    if len(direction_data) == 0 or len(amplitude_data) == 0:
        print("❌ No valid training data found")
        return
    
    # Print data statistics
    print("\n=== Training Data Statistics ===")
    print(f"Direction samples: {len(direction_data)}")
    print(f"Amplitude samples: {len(amplitude_data)}")
    
    # Analyze direction data
    direction_inputs = np.array([d['input'] for d in direction_data])
    direction_targets = np.array([d['target'] for d in direction_data])
    
    print(f"\nDirection Network Input Statistics:")
    print(f"  X positions: min={direction_inputs[:, 0].min():.2f}, max={direction_inputs[:, 0].max():.2f}, mean={direction_inputs[:, 0].mean():.2f}")
    print(f"  Y positions: min={direction_inputs[:, 1].min():.2f}, max={direction_inputs[:, 1].max():.2f}, mean={direction_inputs[:, 1].mean():.2f}")
    print(f"  Current angles (deg): min={np.degrees(direction_inputs[:, 2]).min():.2f}, max={np.degrees(direction_inputs[:, 2]).max():.2f}")
    print(f"  Target angles (deg): min={np.degrees(direction_targets).min():.2f}, max={np.degrees(direction_targets).max():.2f}")
    
    # Analyze amplitude data
    amplitude_inputs = np.array([d['input'] for d in amplitude_data])
    amplitude_targets = np.array([d['target'] for d in amplitude_data])
    
    print(f"\nAmplitude Network Input Statistics:")
    print(f"  X positions: min={amplitude_inputs[:, 0].min():.2f}, max={amplitude_inputs[:, 0].max():.2f}, mean={amplitude_inputs[:, 0].mean():.2f}")
    print(f"  Y positions: min={amplitude_inputs[:, 1].min():.2f}, max={amplitude_inputs[:, 1].max():.2f}, mean={amplitude_inputs[:, 1].mean():.2f}")
    print(f"  Current magnitudes: min={amplitude_inputs[:, 2].min():.2f}, max={amplitude_inputs[:, 2].max():.2f}, mean={amplitude_inputs[:, 2].mean():.2f}")
    print(f"  Closest distances: min={amplitude_inputs[:, 3].min():.2f}, max={amplitude_inputs[:, 3].max():.2f}, mean={amplitude_inputs[:, 3].mean():.2f}")
    print(f"  Moving towards: {np.bincount(amplitude_inputs[:, 4].astype(int))}")
    print(f"  Target magnitudes: min={amplitude_targets.min():.2f}, max={amplitude_targets.max():.2f}, mean={amplitude_targets.mean():.2f}")
    
    # Split data for training
    direction_train, direction_val = train_test_split(direction_data, test_size=0.2, random_state=42)
    amplitude_train, amplitude_val = train_test_split(amplitude_data, test_size=0.2, random_state=42)
    
    # Train direction network
    direction_model = DirectionPredictionNN(input_dim=3).to(device)
    direction_train_losses, direction_val_losses = train_network(
        direction_model, direction_train, direction_val, "Direction Network"
    )
    save_model(direction_model, "direction_model", direction_train_losses, direction_val_losses)
    
    # Train amplitude network
    amplitude_model = AmplitudePredictionNN(input_dim=5).to(device)
    amplitude_train_losses, amplitude_val_losses = train_network(
        amplitude_model, amplitude_train, amplitude_val, "Amplitude Network"
    )
    save_model(amplitude_model, "amplitude_model", amplitude_train_losses, amplitude_val_losses)
    
    print("\n✅ Training completed for both networks!")
    print("Models saved as:")
    print("- direction_model.pth")
    print("- amplitude_model.pth")
    
    # Example trajectory generation
    print("\n=== Example Trajectory Generation ===")
    
    # Get initial state from first track
    first_track = track_df[track_df['trackId'] == 0].iloc[0]
    initial_state = [
        first_track['xCenter'],    # column 4
        first_track['yCenter'],    # column 5
        first_track['xVelocity'],  # column 9
        first_track['yVelocity']   # column 10
    ]
    
    # Generate trajectory
    generated_trajectory = generate_trajectory_with_directional_networks(
        direction_model, amplitude_model, initial_state, track_df, all_trajectories
    )
    
    # Save generated trajectory
    trajectory_df = pd.DataFrame(generated_trajectory)
    trajectory_df.to_csv('generated_directional_trajectory.csv', index=False)
    print("Generated trajectory saved as 'generated_directional_trajectory.csv'")
    """
if __name__ == "__main__":
    main() 