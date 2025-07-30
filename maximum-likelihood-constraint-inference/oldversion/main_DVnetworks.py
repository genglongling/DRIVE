import viz
import pandas
import glob
import numpy as np
import time
import pyglet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--generate_discrete_data", action="store_true")
parser.add_argument("--multi_goal", action="store_true")
parser.add_argument("--visualize", action="store_true")
parser.add_argument("--show_new_demos", action="store_true")
parser.add_argument("--create_gifs", action="store_true")
args = parser.parse_args()

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import random
import torch.optim as optim
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
import hashlib
import json

# continuous state

def smooth_trajectory_data(states, frames, sigma=1.0, window_size=5):
    """
    Smooth trajectory data using Gaussian filtering to reduce noise and spikes
    
    Args:
        states: Array of vehicle states [x, y, vx, vy, ax, ay]
        frames: Array of frame numbers
        sigma: Standard deviation for Gaussian filter (higher = more smoothing)
        window_size: Window size for moving average (alternative smoothing)
    
    Returns:
        smoothed_states: Smoothed state array
    """
    if len(states) < 3:
        return states
    
    smoothed_states = np.copy(states)
    
    # Smooth position (x, y) - use smaller sigma to preserve trajectory shape
    smoothed_states[:, 0] = gaussian_filter1d(states[:, 0], sigma=sigma * 0.5)  # x
    smoothed_states[:, 1] = gaussian_filter1d(states[:, 1], sigma=sigma * 0.5)  # y
    
    # Smooth velocity (vx, vy) - moderate smoothing
    smoothed_states[:, 2] = gaussian_filter1d(states[:, 2], sigma=sigma)  # vx
    smoothed_states[:, 3] = gaussian_filter1d(states[:, 3], sigma=sigma)  # vy
    
    # Smooth acceleration (ax, ay) - more aggressive smoothing to reduce spikes
    smoothed_states[:, 4] = gaussian_filter1d(states[:, 4], sigma=sigma * 1.5)  # ax
    smoothed_states[:, 5] = gaussian_filter1d(states[:, 5], sigma=sigma * 1.5)  # ay
    
    # Recalculate acceleration from smoothed velocity to ensure consistency
    dt = 1.0 / 25.0  # Assuming 25fps
    recalculated_ax = np.gradient(smoothed_states[:, 2], dt)
    recalculated_ay = np.gradient(smoothed_states[:, 3], dt)
    
    # Apply additional smoothing to recalculated acceleration
    smoothed_states[:, 4] = gaussian_filter1d(recalculated_ax, sigma=sigma)
    smoothed_states[:, 5] = gaussian_filter1d(recalculated_ay, sigma=sigma)
    
    return smoothed_states


def save_filtered_transitions(filtered_transitions, csv_file, max_neighbors=5, smoothing_sigma=1.0, 
                            x_threshold=80, y_threshold=-80, frame_rate=25):
    """
    Save filtered transitions to a pickle file with metadata for caching
    
    Args:
        filtered_transitions: List of [features, label] pairs
        csv_file: Original CSV file path
        max_neighbors: Number of neighbors used
        smoothing_sigma: Smoothing parameter used
        x_threshold, y_threshold: Filtering thresholds
        frame_rate: Frame rate used
    """
    # Create cache directory if it doesn't exist
    cache_dir = "./pickles"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a unique filename based on parameters
    params = {
        'csv_file': csv_file,
        'max_neighbors': max_neighbors,
        'smoothing_sigma': smoothing_sigma,
        'x_threshold': x_threshold,
        'y_threshold': y_threshold,
        'frame_rate': frame_rate
    }
    
    # Create hash of parameters for unique filename
    params_str = json.dumps(params, sort_keys=True)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    # Create filename
    csv_basename = os.path.basename(csv_file).replace('.csv', '')
    filename = f"{csv_basename}_filtered_transitions_{params_hash}.pickle"
    filepath = os.path.join(cache_dir, filename)
    
    # Save data with metadata
    data_to_save = {
        'filtered_transitions': filtered_transitions,
        'metadata': {
            'params': params,
            'params_hash': params_hash,
            'num_transitions': len(filtered_transitions),
            'num_positive': sum(1 for t in filtered_transitions if t[1] == 1),
            'num_negative': sum(1 for t in filtered_transitions if t[1] == 0),
            'created_at': pd.Timestamp.now().isoformat()
        }
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"Saved {len(filtered_transitions)} filtered transitions to {filepath}")
    print(f"  Positive samples: {data_to_save['metadata']['num_positive']}")
    print(f"  Negative samples: {data_to_save['metadata']['num_negative']}")
    
    return filepath


def load_filtered_transitions(csv_file, max_neighbors=5, smoothing_sigma=1.0, 
                            x_threshold=80, y_threshold=-80, frame_rate=25, force_reprocess=False):
    """
    Load filtered transitions from cache or reprocess if needed
    
    Args:
        csv_file: CSV file path
        max_neighbors: Number of neighbors
        smoothing_sigma: Smoothing parameter
        x_threshold, y_threshold: Filtering thresholds
        frame_rate: Frame rate
        force_reprocess: Force reprocessing even if cache exists
    
    Returns:
        filtered_transitions: List of [features, label] pairs
    """
    # Create cache directory if it doesn't exist
    cache_dir = "./pickles"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create parameters for filename
    params = {
        'csv_file': csv_file,
        'max_neighbors': max_neighbors,
        'smoothing_sigma': smoothing_sigma,
        'x_threshold': x_threshold,
        'y_threshold': y_threshold,
        'frame_rate': frame_rate
    }
    
    # Create hash and filename
    params_str = json.dumps(params, sort_keys=True)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    csv_basename = os.path.basename(csv_file).replace('.csv', '')
    filename = f"{csv_basename}_filtered_transitions_{params_hash}.pickle"
    filepath = os.path.join(cache_dir, filename)
    
    # Check if cache exists and we're not forcing reprocess
    if not force_reprocess and os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            filtered_transitions = data['filtered_transitions']
            metadata = data['metadata']
            
            print(f"Loaded {len(filtered_transitions)} filtered transitions from cache: {filepath}")
            print(f"  Parameters: max_neighbors={max_neighbors}, smoothing_sigma={smoothing_sigma}")
            print(f"  Positive samples: {metadata['num_positive']}")
            print(f"  Negative samples: {metadata['num_negative']}")
            print(f"  Created: {metadata['created_at']}")
            
            return filtered_transitions
            
        except (pickle.PickleError, KeyError, EOFError) as e:
            print(f"Error loading cache file {filepath}: {e}")
            print("Will reprocess data...")
    
    # If we get here, we need to reprocess
    print(f"Processing data with parameters: max_neighbors={max_neighbors}, smoothing_sigma={smoothing_sigma}")
    filtered_transitions = preprocess_and_filter_trajectories_with_collision_avoidance(
        csv_file, frame_rate, x_threshold, y_threshold, max_neighbors, smoothing_sigma
    )
    
    # Save the processed data
    save_filtered_transitions(filtered_transitions, csv_file, max_neighbors, smoothing_sigma, 
                            x_threshold, y_threshold, frame_rate)
    
    return filtered_transitions


def list_cached_transitions():
    """
    List all cached filtered transition files
    """
    cache_dir = "./pickles"
    if not os.path.exists(cache_dir):
        print("No cache directory found")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('_filtered_transitions_.pickle')]
    
    if not cache_files:
        print("No cached transition files found")
        return
    
    print(f"Found {len(cache_files)} cached transition files:")
    for filename in sorted(cache_files):
        filepath = os.path.join(cache_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            metadata = data['metadata']
            params = metadata['params']
            
            print(f"\n{filename}:")
            print(f"  Transitions: {metadata['num_transitions']}")
            print(f"  Positive: {metadata['num_positive']}, Negative: {metadata['num_negative']}")
            print(f"  Parameters: max_neighbors={params['max_neighbors']}, smoothing_sigma={params['smoothing_sigma']}")
            print(f"  Created: {metadata['created_at']}")
            
        except Exception as e:
            print(f"  Error reading {filename}: {e}")


def clear_transition_cache():
    """
    Clear all cached transition files
    """
    cache_dir = "./pickles"
    if not os.path.exists(cache_dir):
        print("No cache directory found")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('_filtered_transitions_.pickle')]
    
    if not cache_files:
        print("No cached transition files found")
        return
    
    for filename in cache_files:
        filepath = os.path.join(cache_dir, filename)
        try:
            os.remove(filepath)
            print(f"Removed: {filename}")
        except Exception as e:
            print(f"Error removing {filename}: {e}")
    
    print(f"Cleared {len(cache_files)} cached files")


# Preprocess and filter the trajectories from the CSV file
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
        if final_state["xCenter"] < x_threshold and final_state["yCenter"] < y_threshold and (
                first_state["xCenter"] > 160 or first_state["yCenter"] > -40):
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

    print(f"Filtered {len(filtered_transitions)} transitions.")
    return filtered_transitions

def preprocess_and_filter_trajectories_with_collision_avoidance(csv_file, frame_rate=25, x_threshold=80, y_threshold=-80, max_neighbors=5, smoothing_sigma=1.0):
    """
    Preprocess trajectories with collision avoidance features and smoothing
    
    Args:
        csv_file: Path to the tracks CSV file
        frame_rate: Recording frame rate
        x_threshold, y_threshold: Filtering thresholds
        max_neighbors: Maximum number of nearby vehicles to consider
        smoothing_sigma: Standard deviation for Gaussian smoothing (0 = no smoothing)
    """
    df = pd.read_csv(csv_file)
    
    # Group by frame to get all vehicles at each timestamp
    frame_groups = df.groupby('frame')
    
    filtered_transitions = []
    
    # Process data for each unique trackId
    for track_id, group in df.groupby("trackId"):
        # Sort by frame to ensure correct time sequence
        group = group.sort_values("frame")
        
        # Check if the last state satisfies the filtering condition
        final_state = group.iloc[-1]
        first_state = group.iloc[0]
        if final_state["xCenter"] < x_threshold and final_state["yCenter"] < y_threshold and (
                first_state["xCenter"] > 160 or first_state["yCenter"] > -40):
            
            # Extract relevant columns for processing
            states = group[["xCenter", "yCenter", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].values
            frames = group["frame"].values
            
            # Apply smoothing if sigma > 0
            if smoothing_sigma > 0 and len(states) > 3:
                states = smooth_trajectory_data(states, frames, sigma=smoothing_sigma)
            
            # Create transitions (current_state -> next_state)
            for i in range(len(states) - 10):
                current_state = states[i]
                next_state = states[i + 10]
                current_frame = frames[i]
                next_frame = frames[i + 10]
                
                # Get ego vehicle features
                x, y, vx, vy, ax, ay = current_state
                velocity_magnitude = np.sqrt(vx**2 + vy**2)
                velocity_angle = np.arctan2(vy, vx)
                
                # Get other vehicles at current frame
                other_vehicles = get_nearby_vehicles(df, track_id, current_frame, max_neighbors)
                
                # Create collision avoidance features
                collision_features = create_collision_avoidance_features(
                    current_state, next_state, other_vehicles, current_frame, next_frame
                )
                
                # Combine all features
                enhanced_features = [
                    # Ego vehicle current state (6 features)
                    x, y, vx, vy, ax, ay,
                    # Ego vehicle derived features (2 features)
                    velocity_magnitude, velocity_angle,
                    # State transition features (6 features)
                    next_state[0] - x, next_state[1] - y,  # dx, dy
                    next_state[2] - vx, next_state[3] - vy,  # dvx, dvy
                    next_state[4] - ax, next_state[5] - ay,  # dax, day
                ] + collision_features
                
                # Valid transition (positive sample)
                filtered_transitions.append([enhanced_features, 1])
                
                # Create negative samples with collision risk
                collision_negative_samples = create_collision_negative_samples(
                    current_state, other_vehicles, current_frame, max_neighbors
                )
                
                # Create negative samples from data
                data_negative_samples = create_negative_samples_from_data(
                    df, track_id, current_frame, current_state, max_negative_samples=2
                )
                
                # Combine all negative samples
                all_negative_samples = collision_negative_samples + data_negative_samples
                
                for neg_sample in all_negative_samples:
                    filtered_transitions.append([neg_sample, 0])
    
    print(f"Filtered {len(filtered_transitions)} transitions with collision avoidance features.")
    return filtered_transitions

def get_nearby_vehicles(df, ego_track_id, frame, max_neighbors=5):
    """
    Get nearby vehicles at a specific frame
    
    Args:
        df: DataFrame with all vehicle data
        ego_track_id: ID of the ego vehicle
        frame: Current frame number
        max_neighbors: Maximum number of nearby vehicles to return
    
    Returns:
        List of dictionaries with vehicle information
    """
    # Get all vehicles at this frame
    frame_data = df[df['frame'] == frame]
    
    # Get ego vehicle position
    ego_data = frame_data[frame_data['trackId'] == ego_track_id]
    if len(ego_data) == 0:
        return []
    
    ego_x = ego_data['xCenter'].iloc[0]
    ego_y = ego_data['yCenter'].iloc[0]
    
    # Calculate distances to other vehicles
    nearby_vehicles = []
    for _, vehicle in frame_data.iterrows():
        if vehicle['trackId'] != ego_track_id:
            distance = np.sqrt((vehicle['xCenter'] - ego_x)**2 + (vehicle['yCenter'] - ego_y)**2)
            
            vehicle_info = {
                'trackId': vehicle['trackId'],
                'x': vehicle['xCenter'],
                'y': vehicle['yCenter'],
                'vx': vehicle['xVelocity'],
                'vy': vehicle['yVelocity'],
                'ax': vehicle['xAcceleration'],
                'ay': vehicle['yAcceleration'],
                'distance': distance,
                'width': vehicle['width'],
                'length': vehicle['length']
            }
            nearby_vehicles.append(vehicle_info)
    
    # Sort by distance and take closest max_neighbors
    nearby_vehicles.sort(key=lambda x: x['distance'])
    return nearby_vehicles[:max_neighbors]

def create_collision_avoidance_features(current_state, next_state, other_vehicles, current_frame, next_frame):
    """
    Create features for collision avoidance
    
    Args:
        current_state: Current ego vehicle state [x, y, vx, vy, ax, ay]
        next_state: Next ego vehicle state [x, y, vx, vy, ax, ay]
        other_vehicles: List of nearby vehicle dictionaries
        current_frame: Current frame number
        next_frame: Next frame number
    
    Returns:
        List of collision avoidance features
    """
    x, y, vx, vy, ax, ay = current_state
    next_x, next_y, next_vx, next_vy, next_ax, next_ay = next_state
    
    collision_features = []
    
    # If no other vehicles, use default values
    if not other_vehicles:
        # Default features for no collision risk
        collision_features = [
            1000.0,  # min_distance (large value)
            0.0,     # min_time_to_collision (large value)
            0.0,     # collision_risk_score (low risk)
            0.0,     # relative_velocity_towards_ego (no movement)
            0.0,     # closest_vehicle_angle (no vehicle)
            0.0,     # closest_vehicle_velocity_magnitude (no vehicle)
            0.0,     # closest_vehicle_relative_velocity (no vehicle)
            0.0,     # ego_vehicle_velocity_magnitude
            0.0,     # ego_vehicle_acceleration_magnitude
        ]
        return collision_features
    
    # Calculate collision avoidance features
    min_distance = float('inf')
    min_time_to_collision = float('inf')
    collision_risk_score = 0.0
    closest_vehicle_features = [0.0] * 4  # angle, vel_mag, rel_vel, relative_pos
    
    for vehicle in other_vehicles:
        # Current distance
        distance = vehicle['distance']
        min_distance = min(min_distance, distance)
        
        # Relative position vector
        rel_x = vehicle['x'] - x
        rel_y = vehicle['y'] - y
        rel_angle = np.arctan2(rel_y, rel_x)
        rel_pos_magnitude = np.sqrt(rel_x**2 + rel_y**2)  # Always calculate this
        
        # Relative velocity
        rel_vx = vehicle['vx'] - vx
        rel_vy = vehicle['vy'] - vy
        rel_velocity_magnitude = np.sqrt(rel_vx**2 + rel_vy**2)
        
        # Time to collision (simplified)
        if rel_velocity_magnitude > 0.1 and rel_pos_magnitude > 0.1:  # Avoid division by zero
            # Dot product to get velocity towards ego
            velocity_towards_ego = (rel_x * rel_vx + rel_y * rel_vy) / rel_pos_magnitude
            if velocity_towards_ego > 0:  # Moving towards ego
                time_to_collision = distance / velocity_towards_ego
                min_time_to_collision = min(min_time_to_collision, time_to_collision)
        
        # Collision risk score (inverse of distance and time)
        risk_score = 1.0 / (1.0 + distance) * (1.0 / (1.0 + min_time_to_collision))
        collision_risk_score = max(collision_risk_score, risk_score)
        
        # Keep features of closest vehicle
        if distance == min_distance:
            closest_vehicle_features = [
                rel_angle,
                np.sqrt(vehicle['vx']**2 + vehicle['vy']**2),  # velocity magnitude
                rel_velocity_magnitude,
                rel_pos_magnitude
            ]
    
    # Ego vehicle features
    ego_velocity_magnitude = np.sqrt(vx**2 + vy**2)
    ego_acceleration_magnitude = np.sqrt(ax**2 + ay**2)
    
    # Combine all collision avoidance features
    collision_features = [
        min_distance,
        min_time_to_collision if min_time_to_collision != float('inf') else 1000.0,
        collision_risk_score,
        closest_vehicle_features[0],  # closest vehicle angle
        closest_vehicle_features[1],  # closest vehicle velocity magnitude
        closest_vehicle_features[2],  # closest vehicle relative velocity
        closest_vehicle_features[3],  # closest vehicle relative position
        ego_velocity_magnitude,
        ego_acceleration_magnitude
    ]
    
    return collision_features

def create_collision_negative_samples(current_state, other_vehicles, current_frame, max_neighbors):
    """
    Create negative samples that would lead to collisions using multiple strategies
    
    Args:
        current_state: Current ego vehicle state
        other_vehicles: List of nearby vehicles
        current_frame: Current frame number
        max_neighbors: Maximum number of neighbors
    
    Returns:
        List of negative sample feature vectors
    """
    negative_samples = []
    
    if not other_vehicles:
        return negative_samples
    
    x, y, vx, vy, ax, ay = current_state
    velocity_magnitude = np.sqrt(vx**2 + vy**2)
    velocity_angle = np.arctan2(vy, vx)
    
    # Strategy 1: Move directly towards other vehicles
    for vehicle in other_vehicles[:2]:  # Use closest 2 vehicles
        direction_to_vehicle = np.arctan2(vehicle['y'] - y, vehicle['x'] - x)
        
        # Create a transition that moves towards the vehicle
        collision_vx = velocity_magnitude * np.cos(direction_to_vehicle)
        collision_vy = velocity_magnitude * np.sin(direction_to_vehicle)
        collision_x = x + collision_vx * 0.4  # 10 frames at 25fps
        collision_y = y + collision_vy * 0.4
        
        # Calculate collision features for this dangerous transition
        collision_features = create_collision_avoidance_features(
            current_state, 
            [collision_x, collision_y, collision_vx, collision_vy, ax, ay],
            other_vehicles, 
            current_frame, 
            current_frame + 10
        )
        
        # Create feature vector
        negative_features = [
            # Ego vehicle current state (6 features)
            x, y, vx, vy, ax, ay,
            # Ego vehicle derived features (2 features)
            velocity_magnitude, velocity_angle,
            # State transition features (6 features)
            collision_x - x, collision_y - y,  # dx, dy
            collision_vx - vx, collision_vy - vy,  # dvx, dvy
            0, 0,  # dax, day (no acceleration change)
        ] + collision_features
        
        negative_samples.append(negative_features)
    
    # Strategy 2: Physics-violating transitions (too large changes)
    for vehicle in other_vehicles[:1]:  # Use closest vehicle
        # Create unrealistic velocity changes
        unrealistic_vx = vx + np.random.uniform(-10, 10)  # Large velocity change
        unrealistic_vy = vy + np.random.uniform(-10, 10)
        unrealistic_x = x + unrealistic_vx * 0.4
        unrealistic_y = y + unrealistic_vy * 0.4
        
        collision_features = create_collision_avoidance_features(
            current_state, 
            [unrealistic_x, unrealistic_y, unrealistic_vx, unrealistic_vy, ax, ay],
            other_vehicles, 
            current_frame, 
            current_frame + 10
        )
        
        negative_features = [
            x, y, vx, vy, ax, ay,
            velocity_magnitude, velocity_angle,
            unrealistic_x - x, unrealistic_y - y,
            unrealistic_vx - vx, unrealistic_vy - vy,
            0, 0,
        ] + collision_features
        
        negative_samples.append(negative_features)
    
    # Strategy 3: Ignore other vehicles (maintain current trajectory when collision is likely)
    for vehicle in other_vehicles[:1]:
        # If vehicle is close and moving towards ego, but ego continues straight
        if vehicle['distance'] < 20:  # Close vehicle
            # Continue current trajectory (ignoring collision risk)
            ignore_collision_x = x + vx * 0.4
            ignore_collision_y = y + vy * 0.4
            ignore_collision_vx = vx + ax * 0.4
            ignore_collision_vy = vy + ay * 0.4
            
            collision_features = create_collision_avoidance_features(
            current_state, 
            [collision_x, collision_y, collision_vx, collision_vy, ax, ay],
            other_vehicles, 
            current_frame, 
            current_frame + 10
            )
            
            negative_features = [
                x, y, vx, vy, ax, ay,
                velocity_magnitude, velocity_angle,
                ignore_collision_x - x, ignore_collision_y - y,
                ignore_collision_vx - vx, ignore_collision_vy - vy,
                0, 0,
            ] + collision_features
            
            negative_samples.append(negative_features)
    
    # Strategy 4: Sudden lane changes into occupied lanes
    for vehicle in other_vehicles[:1]:
        # Calculate lateral direction to vehicle
        lateral_direction = np.arctan2(vehicle['y'] - y, vehicle['x'] - x)
        
        # Create sudden lateral movement towards vehicle
        sudden_lateral_vx = vx * 0.8  # Reduce forward velocity
        sudden_lateral_vy = velocity_magnitude * np.sin(lateral_direction)  # Move laterally
        sudden_lateral_x = x + sudden_lateral_vx * 0.4
        sudden_lateral_y = y + sudden_lateral_vy * 0.4
        
        collision_features = create_collision_avoidance_features(
            current_state, 
            [sudden_lateral_x, sudden_lateral_y, sudden_lateral_vx, sudden_lateral_vy, ax, ay],
            other_vehicles, 
            current_frame, 
            current_frame + 10
        )
        
        negative_features = [
            x, y, vx, vy, ax, ay,
            velocity_magnitude, velocity_angle,
            sudden_lateral_x - x, sudden_lateral_y - y,
            sudden_lateral_vx - vx, sudden_lateral_vy - vy,
            0, 0,
        ] + collision_features
        
        negative_samples.append(negative_features)
    
    return negative_samples

def create_negative_samples_from_data(df, track_id, current_frame, current_state, max_negative_samples=3):
    """
    Create negative samples from the original trajectory data
    
    Args:
        df: DataFrame with all vehicle data
        track_id: Current vehicle ID
        current_frame: Current frame number
        current_state: Current vehicle state
        max_negative_samples: Maximum number of negative samples to create
    
    Returns:
        List of negative sample feature vectors
    """
    negative_samples = []
    
    # Get the trajectory for this vehicle
    vehicle_trajectory = df[df['trackId'] == track_id].sort_values('frame')
    vehicle_states = vehicle_trajectory[["xCenter", "yCenter", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].values
    vehicle_frames = vehicle_trajectory['frame'].values
    
    # Strategy 1: Use states from different time periods (temporal inconsistency)
    for i in range(min(max_negative_samples, len(vehicle_states) - 10)):
        # Use a state from much later in the trajectory
        future_idx = min(i + 30, len(vehicle_states) - 1)  # 30 frames ahead
        future_state = vehicle_states[future_idx]
        
        # Create features for this temporally inconsistent transition
        x, y, vx, vy, ax, ay = current_state
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        velocity_angle = np.arctan2(vy, vx)
        
        # Get other vehicles at current frame
        other_vehicles = get_nearby_vehicles(df, track_id, current_frame, max_neighbors=5)
        
        # Calculate collision features
        collision_features = create_collision_avoidance_features(
            current_state, future_state, other_vehicles, current_frame, current_frame + 10
        )
        
        # Create feature vector
        negative_features = [
            # Ego vehicle current state (6 features)
            x, y, vx, vy, ax, ay,
            # Ego vehicle derived features (2 features)
            velocity_magnitude, velocity_angle,
            # State transition features (6 features)
            future_state[0] - x, future_state[1] - y,  # dx, dy
            future_state[2] - vx, future_state[3] - vy,  # dvx, dvy
            future_state[4] - ax, future_state[5] - ay,  # dax, day
        ] + collision_features
        
        negative_samples.append(negative_features)
    
    # Strategy 2: Use states from other vehicles (spatial inconsistency)
    other_vehicles_at_frame = df[df['frame'] == current_frame]
    other_vehicle_ids = other_vehicles_at_frame[other_vehicles_at_frame['trackId'] != track_id]['trackId'].unique()
    
    for other_id in other_vehicle_ids[:min(2, len(other_vehicle_ids))]:
        other_vehicle_data = other_vehicles_at_frame[other_vehicles_at_frame['trackId'] == other_id]
        if len(other_vehicle_data) > 0:
            other_state = other_vehicle_data[["xCenter", "yCenter", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].values[0]
            
            # Create features for this spatially inconsistent transition
            x, y, vx, vy, ax, ay = current_state
            velocity_magnitude = np.sqrt(vx**2 + vy**2)
            velocity_angle = np.arctan2(vy, vx)
            
            # Get other vehicles at current frame
            other_vehicles = get_nearby_vehicles(df, track_id, current_frame, max_neighbors=5)
            
                    # Calculate collision features
            collision_features = create_collision_avoidance_features(
                current_state, other_state, other_vehicles, current_frame, current_frame + 10
            )
            
            # Create feature vector
            negative_features = [
                # Ego vehicle current state (6 features)
                x, y, vx, vy, ax, ay,
                # Ego vehicle derived features (2 features)
                velocity_magnitude, velocity_angle,
                # State transition features (6 features)
                other_state[0] - x, other_state[1] - y,  # dx, dy
                other_state[2] - vx, other_state[3] - vy,  # dvx, dvy
                other_state[4] - ax, other_state[5] - ay,  # dax, day
            ] + collision_features
            
            negative_samples.append(negative_features)
    
    # Strategy 3: Perturb real transitions (add noise to make them invalid)
    if len(vehicle_states) > 10:
        # Get the real next state
        current_idx = np.where(vehicle_frames == current_frame)[0]
        if len(current_idx) > 0 and current_idx[0] + 10 < len(vehicle_states):
            real_next_state = vehicle_states[current_idx[0] + 10]
            
            # Add significant noise to make it invalid
            noise_factor = 0.5  # 50% noise
            perturbed_state = real_next_state + np.random.normal(0, noise_factor, 6)
            
            # Create features for this perturbed transition
            x, y, vx, vy, ax, ay = current_state
            velocity_magnitude = np.sqrt(vx**2 + vy**2)
            velocity_angle = np.arctan2(vy, vx)
            
            # Get other vehicles at current frame
            other_vehicles = get_nearby_vehicles(df, track_id, current_frame, max_neighbors=5)
            
                    # Calculate collision features
            collision_features = create_collision_avoidance_features(
                current_state, perturbed_state, other_vehicles, current_frame, current_frame + 10
            )
            
            # Create feature vector
            negative_features = [
                # Ego vehicle current state (6 features)
                x, y, vx, vy, ax, ay,
                # Ego vehicle derived features (2 features)
                velocity_magnitude, velocity_angle,
                # State transition features (6 features)
                perturbed_state[0] - x, perturbed_state[1] - y,  # dx, dy
                perturbed_state[2] - vx, perturbed_state[3] - vy,  # dvx, dvy
                perturbed_state[4] - ax, perturbed_state[5] - ay,  # dax, day
            ] + collision_features
            
            negative_samples.append(negative_features)
    
    return negative_samples


# Define the neural network model (TransitionPredictionNN)
class TransitionPredictionNN(nn.Module):
    def __init__(self, input_dim=23):  # 14 base features + 9 collision features
        super(TransitionPredictionNN, self).__init__()
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


# Load and prepare the dataset
def load_data(filtered_transitions, batch_size=64):
    # Convert the filtered transitions into tensors for training
    input_states = torch.tensor([t[0] for t in filtered_transitions], dtype=torch.float32)
    labels = torch.tensor([t[1] for t in filtered_transitions], dtype=torch.float32)

    # Create a DataLoader
    dataset = TensorDataset(input_states, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader


# Train the model
def train_model(train_loader, epochs=100, lr=0.001):
    model = TransitionPredictionNN(input_dim=23)  # 14 base features + 9 collision features
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features).squeeze()
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return model


# Beam Search for trajectory generation with collision avoidance
def beam_search_with_collision_avoidance(start_state, model, a_values, df, delta_t=0.4, max_depth=100, max_neighbors=5):
    """
    Beam search to generate trajectories with collision avoidance, prioritizing the highest reward.

    Args:
        start_state (list or np.ndarray): Initial state [x, y, vx, vy, ax, ay].
        model (torch.nn.Module): Trained neural network to predict state probabilities.
        a_values (list): Discrete search space for acceleration values.
        df (DataFrame): DataFrame with all vehicle data for collision detection.
        delta_t (float): Time step for state updates.
        max_depth (int): Maximum number of steps to generate trajectory.
        max_neighbors (int): Maximum number of nearby vehicles to consider.

    Returns:
        list: Generated trajectory (sequence of states).
    """
    current_state = start_state.tolist()
    trajectory = [current_state]  # Start trajectory with initial state
    current_frame = 0  # Starting frame

    for step in range(max_depth):
        best_next_state = None
        highest_reward = -float('inf')

        # Get nearby vehicles at current frame (simulate frame progression)
        # In real implementation, you'd get this from the actual data
        other_vehicles = get_nearby_vehicles(df, -1, current_frame, max_neighbors)  # -1 for generated trajectory

        # Generate all possible next states
        for ax_new in a_values:
            for ay_new in a_values:
                # Calculate the next state
                x, y, vx, vy, ax, ay = current_state
                vx_new = vx + ax_new * delta_t
                vy_new = vy + ay_new * delta_t
                x_new = x + vx_new * delta_t
                y_new = y + vy_new * delta_t
                next_state = [x_new, y_new, vx_new, vy_new, ax_new, ay_new]

                # Create enhanced features for evaluation
                # Ego vehicle features
                velocity_magnitude = np.sqrt(vx**2 + vy**2)
                velocity_angle = np.arctan2(vy, vx)
                
                # Create collision avoidance features
                collision_features = create_collision_avoidance_features(
                    current_state, next_state, other_vehicles, current_frame, current_frame + 1
                )
                
                # Create enhanced input features
                enhanced_features = [
                    # Ego vehicle current state (6 features)
                    x, y, vx, vy, ax, ay,
                    # Ego vehicle derived features (2 features)
                    velocity_magnitude, velocity_angle,
                    # State transition features (6 features)
                    next_state[0] - x, next_state[1] - y,  # dx, dy
                    next_state[2] - vx, next_state[3] - vy,  # dvx, dvy
                    next_state[4] - ax, next_state[5] - ay,  # dax, day
                ] + collision_features

                # Evaluate the next state using the model
                state_tensor = torch.tensor(enhanced_features, dtype=torch.float32).unsqueeze(0)
                reward = model(state_tensor).item()

                # Update the best next state if this one has a higher reward
                if reward > highest_reward:
                    best_next_state = next_state
                    highest_reward = reward

        # Append the best state to the trajectory
        trajectory.append(best_next_state)
        current_state = best_next_state
        current_frame += 1

    return trajectory

# Constants
F = 0
n = 35 # dimensionality of state-space
allowed_end_state = [945,946,947,948,980,981,982,983,1015,1016,1017,1018,1050,1051,1052,1053] # [320]
banned_start_state = [1087] # [361]

# Load tracks, tracksMeta, recordingMeta
tracks_files = glob.glob("inD/*_tracks.csv")
tracksMeta_files = glob.glob("inD/*_tracksMeta.csv")
recordingMeta_files = glob.glob("inD/*_recordingMeta.csv")

# Choose the 00_* files
tracks_file, tracksMeta_file, recordingMeta_file = tracks_files[F], tracksMeta_files[F], recordingMeta_files[F]

# Read tracksMeta, recordingsMeta, tracks
tm = pandas.read_csv(tracksMeta_file).to_dict(orient="records")
rm = pandas.read_csv(recordingMeta_file).to_dict(orient="records")
t = pandas.read_csv(tracks_file).groupby(["trackId"], sort=False)

# Normalization
xmin, xmax = np.inf, -np.inf
ymin, ymax = np.inf, -np.inf

bboxes = []
centerpts = []
frames = []
# iterate through groups
for k in range(t.ngroups):

    # Choose the kth track and get lists
    g = t.get_group(k).to_dict(orient="list")

    # Set attributes
    meter_to_px = 1. / rm[0]["orthoPxToMeter"]
    g["xCenterVis"] = np.array(g["xCenter"]) * meter_to_px
    g["yCenterVis"] = -np.array(g["yCenter"]) * meter_to_px
    g["centerVis"] = np.stack([np.array(g["xCenter"]), -np.array(g["yCenter"])], axis=-1) * meter_to_px
    g["widthVis"] = np.array(g["width"]) * meter_to_px
    g["lengthVis"] = np.array(g["length"]) * meter_to_px
    g["headingVis"] = np.array(g["heading"]) * -1
    g["headingVis"][g["headingVis"] < 0] += 360
    g["bboxVis"] = viz.calculate_rotated_bboxes(
        g["xCenterVis"], g["yCenterVis"],
        g["lengthVis"], g["widthVis"],
        np.deg2rad(g["headingVis"])
    )

    # M bounding boxes
    bbox = g["bboxVis"]
    centerpt = g["centerVis"]
    bboxes += [bbox]
    centerpts += [centerpt]
    frames += [g["frame"]]
    xmin, xmax = min(xmin, np.min(bbox[:, :, 0])), max(xmax, np.max(bbox[:, :, 0]))
    ymin, ymax = min(ymin, np.min(bbox[:, :, 1])), max(ymax, np.max(bbox[:, :, 1]))

# normalize
for i in range(len(bboxes)):
    bboxes[i][:, :, 0] = (bboxes[i][:, :, 0]-xmin) / (xmax-xmin) * 1000.
    bboxes[i][:, :, 1] = (bboxes[i][:, :, 1]-ymin) / (ymax-ymin) * 1000.
    centerpts[i][:, 0] = (centerpts[i][:, 0]-xmin) / (xmax-xmin) * 1000.
    centerpts[i][:, 1] = (centerpts[i][:, 1]-ymin) / (ymax-ymin) * 1000.

# See if there is a constraints.pickle
try:
    import pickle, os
    if not os.path.exists("pickles"):
        os.mkdir("pickles")
    with open('pickles/constraints.pickle', 'rb') as handle:
        constraints = pickle.load(handle)
except:
    print("\nNo constraints.pickle! Simulation rendering will not show constraints")


class DiscreteGrid(viz.Group):
    def __init__(self, x, y, w, h, arr):
        self.arr = arr
        self.itemsarr = np.array([[None for j in range(arr.shape[1])] for i in range(arr.shape[0])])
        self.allpts = [[None for j in range(arr.shape[1])] for i in range(arr.shape[0])]
        self.xsize, self.ysize = w/arr.shape[0], h/arr.shape[1]
        self.colors = {0:(0,0,0,0.5), 1:(1,0,0,0.5), 2:(0,1,0,0.5), 3:(0,0,1,0.5)}
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                pts = [[x+i*self.xsize+self.xsize/10, y+j*self.ysize+self.ysize/10],
                       [x+(i+1)*self.xsize-self.xsize/10, y+j*self.ysize+self.ysize/10],
                       [x+(i+1)*self.xsize-self.xsize/10, y+(j+1)*self.ysize-self.ysize/10],
                       [x+i*self.xsize+self.xsize/10, y+(j+1)*self.ysize-self.ysize/10]]
                self.allpts[i][j] = pts
                self.itemsarr[i][j] = viz.Rectangle(pts, color = self.colors[arr[i][j]])
        try:
            for pt in constraints["state"]:
                self.itemsarr[pt%n][pt//n].color = (1, 1, 1, 1)
        except:
            pass
        super().__init__(items = self.itemsarr.flatten().tolist())

# Draw Canvas
canvas = viz.Canvas(1000, 1000, id = "000")
canvas.set_visible(False)
pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
arr = np.zeros((n, n))
canvas.items += [DiscreteGrid(20, 60, 1000-30, 1000-60, arr)]

def localize(x, y, grid):
    for i in range(len(grid.allpts)):
        for j in range(len(grid.allpts[0])):
            pt1, pt2, pt3, pt4 = grid.allpts[i][j]
            x1, x2 = pt1[0] - grid.xsize/10, pt2[0] + grid.xsize/10
            y1, y2 = pt2[1] - grid.ysize/10, pt3[1] + grid.ysize/10
            if x1 <= x <= x2 and y1 <= y <= y2:
                return (i, j)
    return (-1, -1)

def delocalize(pt, grid):
    for i in range(len(grid.allpts)):
        for j in range(len(grid.allpts[0])):
            pt1, pt2, pt3, pt4 = grid.allpts[i][j]
            x1, x2 = pt1[0] - grid.xsize/10, pt2[0] + grid.xsize/10
            y1, y2 = pt2[1] - grid.ysize/10, pt3[1] + grid.ysize/10
            if i+j*n == pt:
                return np.array([(x1+x2)/2, (y1+y2)/2])



def analyze_smoothing_effect(csv_file, track_id=0, sigma_values=[0, 0.5, 1.0, 1.5, 2.0]):
    """
    Analyze the effect of different smoothing parameters on trajectory data
    
    Args:
        csv_file: Path to the tracks CSV file
        track_id: Track ID to analyze
        sigma_values: List of sigma values to test
    """
    df = pd.read_csv(csv_file)
    track_data = df[df['trackId'] == track_id].sort_values('frame')
    
    if len(track_data) == 0:
        print(f"Track {track_id} not found")
        return
    
    states = track_data[["xCenter", "yCenter", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].values
    frames = track_data["frame"].values
    
    print(f"Analyzing smoothing effect for track {track_id} with {len(states)} frames")
    print(f"Original data statistics:")
    print(f"  Position std: x={np.std(states[:, 0]):.2f}, y={np.std(states[:, 1]):.2f}")
    print(f"  Velocity std: vx={np.std(states[:, 2]):.2f}, vy={np.std(states[:, 3]):.2f}")
    print(f"  Acceleration std: ax={np.std(states[:, 4]):.2f}, ay={np.std(states[:, 5]):.2f}")
    
    for sigma in sigma_values:
        if sigma == 0:
            smoothed_states = states
            label = "Original"
        else:
            smoothed_states = smooth_trajectory_data(states, frames, sigma=sigma)
            label = f"σ={sigma}"
        
        # Calculate noise reduction
        pos_noise_reduction = 1 - (np.std(smoothed_states[:, :2]) / np.std(states[:, :2]))
        vel_noise_reduction = 1 - (np.std(smoothed_states[:, 2:4]) / np.std(states[:, 2:4]))
        acc_noise_reduction = 1 - (np.std(smoothed_states[:, 4:6]) / np.std(states[:, 4:6]))
        
        print(f"\n{label}:")
        print(f"  Position noise reduction: {pos_noise_reduction*100:.1f}%")
        print(f"  Velocity noise reduction: {vel_noise_reduction*100:.1f}%")
        print(f"  Acceleration noise reduction: {acc_noise_reduction*100:.1f}%")
        print(f"  Max acceleration change: {np.max(np.abs(smoothed_states[:, 4:6])):.2f}")



def analyze_smoothing_effect(csv_file, track_id=0, sigma_values=[0, 0.5, 1.0, 1.5, 2.0]):
    """
    Analyze the effect of different smoothing parameters on trajectory data
    
    Args:
        csv_file: Path to the tracks CSV file
        track_id: Track ID to analyze
        sigma_values: List of sigma values to test
    """
    df = pd.read_csv(csv_file)
    track_data = df[df['trackId'] == track_id].sort_values('frame')
    
    if len(track_data) == 0:
        print(f"Track {track_id} not found")
        return
    
    states = track_data[["xCenter", "yCenter", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].values
    frames = track_data["frame"].values
    
    print(f"Analyzing smoothing effect for track {track_id} with {len(states)} frames")
    print(f"Original data statistics:")
    print(f"  Position std: x={np.std(states[:, 0]):.2f}, y={np.std(states[:, 1]):.2f}")
    print(f"  Velocity std: vx={np.std(states[:, 2]):.2f}, vy={np.std(states[:, 3]):.2f}")
    print(f"  Acceleration std: ax={np.std(states[:, 4]):.2f}, ay={np.std(states[:, 5]):.2f}")
    
    for sigma in sigma_values:
        if sigma == 0:
            smoothed_states = states
            label = "Original"
        else:
            smoothed_states = smooth_trajectory_data(states, frames, sigma=sigma)
            label = f"σ={sigma}"
        
        # Calculate noise reduction
        pos_noise_reduction = 1 - (np.std(smoothed_states[:, :2]) / np.std(states[:, :2]))
        vel_noise_reduction = 1 - (np.std(smoothed_states[:, 2:4]) / np.std(states[:, 2:4]))
        acc_noise_reduction = 1 - (np.std(smoothed_states[:, 4:6]) / np.std(states[:, 4:6]))
        
        print(f"\n{label}:")
        print(f"  Position noise reduction: {pos_noise_reduction*100:.1f}%")
        print(f"  Velocity noise reduction: {vel_noise_reduction*100:.1f}%")
        print(f"  Acceleration noise reduction: {acc_noise_reduction*100:.1f}%")
        print(f"  Max acceleration change: {np.max(np.abs(smoothed_states[:, 4:6])):.2f}")


import matplotlib.pyplot as plt
import numpy as np

# Combine dataset and demo trajectories for plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the dataset trajectory (center points)
for demo_pts in centerpts:
    demo_pts = np.array(demo_pts)
    #ax.plot(demo_pts[:, 0], demo_pts[:, 1], marker='x', color='yellow')

# If constraints exist, plot them as red triangles with x, y values
if constraints and "state" in constraints:
    pt = []
    for state_value in constraints["state"]:
        # print(state_value)
        # # Use state_value as the x-coordinate and idx as the y-coordinate (index in the list)
        pt += [delocalize(state_value, canvas.items[-1])]   # x-coordinate
    pt = np.array(pt)
    ax.plot(pt[:, 0], pt[:, 1], 'rx', markersize=10)

# finish the trajectories plotting by beam search

a_values = [-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2]  # Discrete acceleration values
traj_list = []
csv_file = "./inD/00_tracks.csv"  # Replace with the actual CSV file

# Analyze smoothing effect first
print("=== Analyzing Smoothing Effect ===")
analyze_smoothing_effect(csv_file, track_id=0, sigma_values=[0, 0.5, 1.0, 1.5, 2.0])

# Load or process filtered transitions with caching
print("\n=== Loading/Processing Filtered Transitions ===")
filtered_transitions = load_filtered_transitions(
    csv_file, 
    max_neighbors=5, 
    smoothing_sigma=1.0,
    force_reprocess=False  # Set to True to force reprocessing
)

train_loader = load_data(filtered_transitions)
model, train_losses, val_losses = train_model(train_loader, epochs=200, lr=0.001)

for i in range(1000):
    # CHANGE -> Extract start state for each sample -
    start_state = train_loader.dataset[i][0][:6].numpy()

    # Generate the trajectory using Beam Search
    best_trajectory = beam_search_with_collision_avoidance(start_state, model, a_values, pandas.read_csv(csv_file), max_depth=100)

    # Recalculate xmin and ymin for the specific trajectory
    trajectory_x = [state[0] for state in best_trajectory]  # Extract all x values
    trajectory_y = [state[1] for state in best_trajectory]  # Extract all y values

    xmin_local = min(trajectory_x)
    xmax_local = max(trajectory_x)
    ymin_local = min(trajectory_y)
    ymax_local = max(trajectory_y)

    # Normalize trajectory to demo points format
    normalized_trajectory = []
    for state in best_trajectory:
        x, y, vx, vy, a_x, a_y = state

        # Normalize x and y using the local min and max
        x_normalized = (x - xmin_local) / (xmax_local - xmin_local) * 1000
        y_normalized = (y - ymin_local) / (ymax_local - ymin_local) * 1000

        # Replace original x and y with normalized values
        normalized_state = [x_normalized, y_normalized, vx, vy, a_x, a_y]
        normalized_trajectory.append(normalized_state)

    # Append normalized trajectory to the trajectory list
    traj_list.append(normalized_trajectory)

# Output all trajectories to a single CSV file with trajectory number
import csv
output_path = "beamsearch_trajectories_all.csv"
with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["traj_id", "x", "y", "vx", "vy", "ax", "ay"])
    for traj_id, traj in enumerate(traj_list, start=1):
        for state in traj:
            writer.writerow([traj_id] + list(state))

for i, traj in enumerate(traj_list):
    traj = np.array(traj)  # Convert to numpy array if not already
    ax.plot(traj[:, 0], traj[:, 1], marker='o') #label=f"Trajectory {i+1}

# Customize the plot

ax.set_title("Beam Search Trajectories",fontsize=24, color='white')  
ax.set_xlabel("X", fontsize=18, color='white')  
ax.set_ylabel("Y", fontsize=18, color='white')  
ax.legend(fontsize=14, facecolor='white', edgecolor='white', labelcolor='white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')       
ax.grid(True)
# Or set individually:
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')

# Save the combined plot
output_path = "./visualization/trajectory_beamsearch.png"
plt.savefig(output_path, dpi=300, transparent=True)
plt.close()  # Close the plot to free memory
print(f"Visualization of demo and dataset trajectories saved as {output_path}")


import matplotlib.pyplot as plt
import numpy as np

# Limit to 100 trajectories
num_traj = 10
selected_traj = traj_list[:num_traj]

# Prepare data for plotting
all_time = []
all_velocity = []
all_acceleration = []

for traj_idx, traj in enumerate(selected_traj):
    time_steps = np.arange(len(traj))  # Time steps (assume uniform time intervals)
    vx = [state[2] for state in traj]  # Extract vx
    vy = [state[3] for state in traj]  # Extract vy
    ax = [state[4] for state in traj]  # Extract ax
    ay = [state[5] for state in traj]  # Extract ay

    # Combine for plotting
    all_time.append(time_steps)
    all_velocity.append((vx, vy))
    all_acceleration.append((ax, ay))

# Plot velocity and acceleration
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Velocity Plot
for traj_idx, (time, (vx, vy)) in enumerate(zip(all_time, all_velocity)):
    axes[0].plot(time, vx, label=f"Traj {traj_idx+1} - vx", alpha=0.7)
    axes[0].plot(time, vy, label=f"Traj {traj_idx+1} - vy", linestyle="dashed", alpha=0.7)

axes[0].set_title("Velocity Change Over Time")
axes[0].set_ylabel("Velocity (vx, vy)")
axes[0].legend(loc="upper right", fontsize=8, ncol=2)

# Save the velocity plot
velocity_plot_path = "./visualization/velocity_beamsearch.png"
fig.savefig(velocity_plot_path)

# Acceleration Plot
for traj_idx, (time, (ax, ay)) in enumerate(zip(all_time, all_acceleration)):
    axes[1].plot(time, ax, label=f"Traj {traj_idx+1} - ax", alpha=0.7)
    axes[1].plot(time, ay, label=f"Traj {traj_idx+1} - ay", linestyle="dashed", alpha=0.7)

axes[1].set_title("Acceleration Change Over Time")
axes[1].set_xlabel("Time Steps")
axes[1].set_ylabel("Acceleration (ax, ay)")
axes[1].legend(loc="upper right", fontsize=8, ncol=2)

# Save the acceleration plot
acceleration_plot_path = "./visualization/acceleration_beamsearch.png"
fig.savefig(acceleration_plot_path)

plt.tight_layout()
plt.show()


def analyze_collision_avoidance_features(filtered_transitions):
    """
    Analyze collision avoidance features in the training data
    """
    print("\n=== Collision Avoidance Feature Analysis ===")
    
    # Extract collision features (last 9 features)
    collision_features = np.array([t[0][-9:] for t in filtered_transitions])
    labels = np.array([t[1] for t in filtered_transitions])
    
    feature_names = [
        'min_distance', 'min_time_to_collision', 'collision_risk_score',
        'closest_vehicle_angle', 'closest_vehicle_velocity_magnitude',
        'closest_vehicle_relative_velocity', 'closest_vehicle_relative_position',
        'ego_velocity_magnitude', 'ego_acceleration_magnitude'
    ]
    
    print(f"Collision avoidance feature statistics:")
    for i, name in enumerate(feature_names):
        feature_data = collision_features[:, i]
        positive_data = feature_data[labels == 1]
        negative_data = feature_data[labels == 0]
        
        print(f"\n{name}:")
        print(f"  All samples: mean={np.mean(feature_data):.3f}, std={np.std(feature_data):.3f}")
        print(f"  Positive samples: mean={np.mean(positive_data):.3f}, std={np.std(positive_data):.3f}")
        print(f"  Negative samples: mean={np.mean(negative_data):.3f}, std={np.std(negative_data):.3f}")
    
    # Analyze collision risk distribution
    collision_risk = collision_features[:, 2]  # collision_risk_score
    print(f"\nCollision Risk Analysis:")
    print(f"  Low risk (0-0.1): {np.sum(collision_risk < 0.1)} samples")
    print(f"  Medium risk (0.1-0.5): {np.sum((collision_risk >= 0.1) & (collision_risk < 0.5))} samples")
    print(f"  High risk (0.5+): {np.sum(collision_risk >= 0.5)} samples")
    
    # Analyze distance distribution
    min_distances = collision_features[:, 0]  # min_distance
    print(f"\nDistance Analysis:")
    print(f"  Very close (<10m): {np.sum(min_distances < 10)} samples")
    print(f"  Close (10-50m): {np.sum((min_distances >= 10) & (min_distances < 50))} samples")
    print(f"  Far (50m+): {np.sum(min_distances >= 50)} samples")

def analyze_training_data(filtered_transitions):
    """
    Analyze the quality of training data
    """
    print("\n=== Training Data Analysis ===")
    
    # Count positive and negative samples
    positive_samples = sum(1 for t in filtered_transitions if t[1] == 1)
    negative_samples = sum(1 for t in filtered_transitions if t[1] == 0)
    total_samples = len(filtered_transitions)
    
    print(f"Total samples: {total_samples}")
    print(f"Positive samples: {positive_samples} ({positive_samples/total_samples*100:.1f}%)")
    print(f"Negative samples: {negative_samples} ({negative_samples/total_samples*100:.1f}%)")
    
    # Analyze feature distributions
    features = np.array([t[0] for t in filtered_transitions])
    labels = np.array([t[1] for t in filtered_transitions])
    
    print(f"\nFeature statistics:")
    print(f"Feature shape: {features.shape}")
    print(f"Mean values: {np.mean(features, axis=0)}")
    print(f"Std values: {np.std(features, axis=0)}")
    
    # Check for class imbalance
    if positive_samples / negative_samples > 2 or negative_samples / positive_samples > 2:
        print(f"⚠️  Warning: Class imbalance detected!")
        print(f"   Consider using class weights or balanced sampling")
    
    # Analyze collision avoidance features
    analyze_collision_avoidance_features(filtered_transitions)


# Command-line interface for cache management
if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "list":
            print("=== Listing Cached Transitions ===")
            list_cached_transitions()
            sys.exit(0)
        elif command == "clear":
            print("=== Clearing Transition Cache ===")
            clear_transition_cache()
            sys.exit(0)
        elif command == "reprocess":
            print("=== Forcing Reprocessing ===")
            force_reprocess = True
        else:
            print(f"Unknown command: {command}")
            print("Available commands: list, clear, reprocess")
            print("Usage: python main_DVnetworks.py [list|clear|reprocess]")
            sys.exit(1)
    else:
        force_reprocess = False
    
    # Set up the environment
    traj_list = []
    csv_file = "./inD/00_tracks.csv"  # Replace with the actual CSV file
    
    # Analyze smoothing effect first
    print("=== Analyzing Smoothing Effect ===")
    analyze_smoothing_effect(csv_file, track_id=0, sigma_values=[0, 0.5, 1.0, 1.5, 2.0])
    
    # Load or process filtered transitions with caching
    print("\n=== Loading/Processing Filtered Transitions ===")
    filtered_transitions = load_filtered_transitions(
        csv_file, 
        max_neighbors=5, 
        smoothing_sigma=1.0,
        force_reprocess=force_reprocess
    )
    
    train_loader = load_data(filtered_transitions)
    model = train_model(train_loader, epochs=200, lr=0.001)
