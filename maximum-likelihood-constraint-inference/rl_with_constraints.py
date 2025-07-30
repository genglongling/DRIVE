import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import random
import os
import pickle
import json
from typing import List, Tuple, Dict, Optional

# Import your existing constraint model
from constraint_model_utils import TransitionPredictionNN, load_model, get_nearby_vehicles, create_collision_avoidance_features

class ConstraintAwarePolicyNetwork(nn.Module):
    """
    Policy network that predicts optimal actions while respecting learned constraints.
    
    This network takes the current state and outputs action probabilities,
    but uses a constraint model to mask out invalid actions.
    """
    def __init__(self, state_dim=23, action_dim=4, hidden_dim=128, constraint_model=None):
        super(ConstraintAwarePolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_model = constraint_model
        
        # Policy network architecture
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network for critic (optional, for actor-critic methods)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state, constraint_mask=None):
        """
        Forward pass through the policy network.
        
        Args:
            state: Current state tensor [batch_size, state_dim]
            constraint_mask: Optional mask for invalid actions [batch_size, action_dim]
        
        Returns:
            action_logits: Raw action logits
            action_probs: Action probabilities (with constraints applied)
            value: State value estimate
        """
        # Get raw action logits
        action_logits = self.policy_net(state)
        value = self.value_net(state)
        
        # Apply constraint mask if provided
        if constraint_mask is not None:
            action_logits = action_logits.masked_fill(constraint_mask == 0, -1e9)
        
        # Convert to probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_logits, action_probs, value
    
    def get_constraint_mask(self, state, action_space):
        """
        Generate constraint mask using the learned constraint model.
        
        Args:
            state: Current state [batch_size, state_dim]
            action_space: List of possible actions to evaluate
        
        Returns:
            constraint_mask: Binary mask indicating valid actions [batch_size, action_dim]
        """
        if self.constraint_model is None:
            return torch.ones(state.shape[0], len(action_space))
        
        constraint_mask = torch.ones(state.shape[0], len(action_space))
        
        with torch.no_grad():
            for i, action in enumerate(action_space):
                # Create next state by applying action
                next_states = self.apply_action_to_state(state, action)
                
                # Evaluate constraint satisfaction
                constraint_scores = self.constraint_model(next_states).squeeze()
                valid_actions = (constraint_scores > 0.5).float()  # Threshold for valid transitions
                
                # Ensure valid_actions has the right shape
                if valid_actions.dim() == 0:  # Scalar
                    valid_actions = valid_actions.unsqueeze(0)
                
                constraint_mask[:, i] = valid_actions
        
        return constraint_mask
    
    def apply_action_to_state(self, state, action):
        """
        Apply action to current state to get next state.
        This is a simplified physics model - you may want to use your existing dynamics.
        
        Args:
            state: Current state [batch_size, state_dim]
            action: Action to apply [ax, ay] (acceleration components)
        
        Returns:
            next_state: Predicted next state
        """
        # Extract state components (assuming format from your constraint model)
        x, y, vx, vy, ax, ay = state[:, 1:7].T  # Skip track_id, get position, velocity, acceleration
        
        # Apply action (acceleration)
        dt = 0.4  # Time step (from your beam search)
        new_ax, new_ay = action[0], action[1]
        
        # Update velocity and position using simple physics
        new_vx = vx + new_ax * dt
        new_vy = vy + new_ay * dt
        new_x = x + vx * dt + 0.5 * new_ax * dt**2
        new_y = y + vy * dt + 0.5 * new_ay * dt**2
        
        # Create next state (maintain same format as input)
        next_state = state.clone()
        next_state[:, 1] = new_x  # x
        next_state[:, 2] = new_y  # y
        next_state[:, 3] = new_vx  # vx
        next_state[:, 4] = new_vy  # vy
        next_state[:, 5] = new_ax  # ax
        next_state[:, 6] = new_ay  # ay
        
        return next_state


class ConstraintGuidedRL:
    """
    Main RL class that integrates constraint learning with policy optimization.
    """
    def __init__(self, state_dim=23, action_dim=9, constraint_model_path=None, 
                 learning_rate=0.001, gamma=0.99, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        
        # Load constraint model if provided
        self.constraint_model = None
        if constraint_model_path and os.path.exists(constraint_model_path):
            print(f"Loading constraint model from {constraint_model_path}")
            try:
                self.constraint_model, metadata = load_model(constraint_model_path)
                self.constraint_model.eval()
                print(f"✓ Constraint model loaded successfully")
                print(f"  Model config: {metadata.get('model_config', 'Unknown')}")
            except Exception as e:
                print(f"✗ Error loading constraint model: {e}")
                print("  Proceeding without constraint model...")
                self.constraint_model = None
        elif constraint_model_path:
            print(f"✗ Constraint model path provided but file not found: {constraint_model_path}")
            print("  Proceeding without constraint model...")
        else:
            print("  No constraint model path provided, proceeding without constraint model...")
        
        # Test constraint model if loaded
        if self.constraint_model is not None:
            try:
                # Test with dummy input
                test_input = torch.randn(1, 23)  # 23 features
                with torch.no_grad():
                    test_output = self.constraint_model(test_input)
                print(f"✓ Constraint model test successful (output shape: {test_output.shape})")
            except Exception as e:
                print(f"✗ Constraint model test failed: {e}")
                print("  Proceeding without constraint model...")
                self.constraint_model = None
        
                # Initialize policy network
        self.policy_net = ConstraintAwarePolicyNetwork(
            state_dim=state_dim,
            action_dim=9,  # Fixed to match action space size
            constraint_model=self.constraint_model
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Action space (acceleration components)
        self.action_space = [
            [-2.0, -2.0], [-2.0, 0.0], [-2.0, 2.0],
            [0.0, -2.0],  [0.0, 0.0],  [0.0, 2.0],
            [2.0, -2.0],  [2.0, 0.0],  [2.0, 2.0]
        ]
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'constraint_violations': [],
            'policy_losses': [],
            'value_losses': [],
            # Travel time optimization metrics
            'travel_times': [],
            'goal_distances': [],
            'velocity_violations': [],
            'acceleration_violations': [],
            'collision_risks': [],
            'constraint_satisfaction': [],
            # Trajectory data collection
            'trajectory_data': []  # List of trajectory dictionaries
        }
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy with constraint awareness.
        
        Args:
            state: Current state
            training: Whether in training mode
        
        Returns:
            action: Selected action index
            action_probs: Action probabilities
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get constraint mask
        constraint_mask = self.policy_net.get_constraint_mask(state_tensor, self.action_space)
        
        # Get action probabilities
        _, action_probs, _ = self.policy_net(state_tensor, constraint_mask)
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            # Random action among valid actions
            valid_actions = constraint_mask.squeeze().nonzero().squeeze()
            if len(valid_actions) > 0:
                action_idx = random.choice(valid_actions.tolist())
            else:
                # If no valid actions, generate a constraint-satisfying action
                action_idx = self._generate_constraint_satisfying_action(state)
        else:
            # Greedy action - ensure it's valid
            action_idx = action_probs.argmax().item()
            if constraint_mask.squeeze()[action_idx] == 0:
                # If greedy action is invalid, generate a constraint-satisfying action
                action_idx = self._generate_constraint_satisfying_action(state)
        
        return action_idx, action_probs.squeeze()
    
    def _generate_constraint_satisfying_action(self, state):
        """
        Generate a constraint-satisfying action when all actions violate constraints.
        
        Args:
            state: Current state
        
        Returns:
            action_idx: Index of a constraint-satisfying action
        """
        # Extract current velocity and acceleration
        vx, vy = state[3], state[4]
        ax, ay = state[5], state[6]
        
        # Check if current state already violates constraints
        current_speed = np.sqrt(vx**2 + vy**2)
        current_accel = np.sqrt(ax**2 + ay**2)
        
        # If current state violates constraints, generate aggressive deceleration actions
        if current_speed > 30.0 or current_accel > 5.0:
            # Generate aggressive deceleration actions to quickly reduce violations
            aggressive_deceleration_actions = [
                [-2.0, 0.0],   # Strong deceleration in x
                [0.0, -2.0],   # Strong deceleration in y
                [-2.0, -2.0],  # Strong deceleration in both directions
                [-3.0, 0.0],   # Very strong deceleration in x
                [0.0, -3.0],   # Very strong deceleration in y
                [-1.5, -1.5],  # Moderate deceleration in both directions
            ]
            
            # Test each aggressive deceleration action
            for action in aggressive_deceleration_actions:
                next_state = self._apply_action_to_state(state, action)
                if self._is_action_valid(state, next_state):
                    action_idx = self._find_closest_action(action)
                    return action_idx
        
        # Generate conservative actions that are likely to satisfy constraints
        conservative_actions = [
            [0.0, 0.0],   # No acceleration
            [0.5, 0.0],   # Small positive x acceleration
            [0.0, 0.5],   # Small positive y acceleration
            [-0.5, 0.0],  # Small negative x acceleration
            [0.0, -0.5],  # Small negative y acceleration
        ]
        
        # Test each conservative action
        for action in conservative_actions:
            # Apply action to get next state
            next_state = self._apply_action_to_state(state, action)
            
            # Check if this action satisfies constraints
            if self._is_action_valid(state, next_state):
                # Find the closest action in our action space
                action_idx = self._find_closest_action(action)
                return action_idx
        
        # If no action works, return the safest action (no acceleration)
        return 4  # Index of [0.0, 0.0] in action space
    
    def _apply_action_to_state(self, state, action):
        """
        Apply action to state to get next state.
        
        Args:
            state: Current state
            action: Action [ax, ay]
        
        Returns:
            next_state: Next state after applying action
        """
        dt = 0.4  # Time step
        
        # Extract current state components
        x, y = state[1], state[2]
        vx, vy = state[3], state[4]
        ax, ay = state[5], state[6]
        
        # Apply action
        new_ax, new_ay = action[0], action[1]
        
        # Update velocity and position
        new_vx = vx + new_ax * dt
        new_vy = vy + new_ay * dt
        new_x = x + vx * dt + 0.5 * new_ax * dt**2
        new_y = y + vy * dt + 0.5 * new_ay * dt**2
        
        # Create next state
        next_state = state.copy()
        next_state[1] = new_x
        next_state[2] = new_y
        next_state[3] = new_vx
        next_state[4] = new_vy
        next_state[5] = new_ax
        next_state[6] = new_ay
        
        return next_state
    
    def _is_action_valid(self, state, next_state):
        """
        Check if an action leads to a valid (constraint-satisfying) state.
        
        Args:
            state: Current state
            next_state: Next state after applying action
        
        Returns:
            valid: True if action satisfies constraints
        """
        # Extract velocity and acceleration from next state
        vx, vy = next_state[3], next_state[4]
        ax, ay = next_state[5], next_state[6]
        
        # Check velocity constraint
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 30.0:  # Maximum velocity
            return False
        
        # Check acceleration constraint
        accel = np.sqrt(ax**2 + ay**2)
        if accel > 5.0:  # Maximum acceleration
            return False
        
        # Check bounds (simple boundary check)
        x, y = next_state[1], next_state[2]
        if x < 0 or x > 1000 or y < 0 or y > 1000:
            return False
        
        return True
    
    def _find_closest_action(self, action):
        """
        Find the closest action in the action space to the given action.
        
        Args:
            action: Action [ax, ay]
        
        Returns:
            action_idx: Index of closest action in action space
        """
        min_distance = float('inf')
        closest_idx = 4  # Default to [0.0, 0.0]
        
        for i, space_action in enumerate(self.action_space):
            distance = np.sqrt((action[0] - space_action[0])**2 + (action[1] - space_action[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        return closest_idx
    
    def _normalize_state_if_needed(self, state):
        """
        Normalize state to satisfy constraints if it violates them.
        
        Args:
            state: Current state
        
        Returns:
            normalized_state: State that satisfies constraints
        """
        # Extract velocity and acceleration
        vx, vy = state[3], state[4]
        ax, ay = state[5], state[6]
        
        # Check if state violates constraints
        speed = np.sqrt(vx**2 + vy**2)
        accel = np.sqrt(ax**2 + ay**2)
        
        normalized_state = state.copy()
        
        # Normalize velocity if it exceeds limits
        if speed > 30.0:
            # Scale down velocity to maximum allowed
            scale_factor = 30.0 / speed
            normalized_state[3] = vx * scale_factor
            normalized_state[4] = vy * scale_factor
            print(f"Normalized velocity from {speed:.1f} to {30.0}")
        
        # Normalize acceleration if it exceeds limits
        if accel > 5.0:
            # Scale down acceleration to maximum allowed
            scale_factor = 5.0 / accel
            normalized_state[5] = ax * scale_factor
            normalized_state[6] = ay * scale_factor
            print(f"Normalized acceleration from {accel:.1f} to {5.0}")
        
        return normalized_state
    
    def convert_state_for_constraint_model(self, state, df=None, frame=None):
        """
        Convert RL state to format expected by constraint model.
        
        Args:
            state: RL state [track_id, x, y, vx, vy, ax, ay, lon_vel, lat_vel, lon_acc, lat_acc, frame, ...]
            df: DataFrame with other vehicles
            frame: Current frame number
        
        Returns:
            constraint_state: State in format expected by constraint model [23 features]
        """
        # Extract base features (same as training data)
        x, y = state[1], state[2]
        vx, vy = state[3], state[4]
        ax, ay = state[5], state[6]
        lon_vel, lat_vel = state[7], state[8]
        lon_acc, lat_acc = state[9], state[10]
        frame_num = state[11]
        
        # Create collision avoidance features
        collision_features = [0.0] * 9  # Default values
        
        if df is not None and frame is not None:
            # Get nearby vehicles
            other_vehicles = get_nearby_vehicles(df, -1, frame, max_neighbors=5)
            
            if other_vehicles:
                # Calculate collision features
                min_distance = float('inf')
                min_time_to_collision = float('inf')
                collision_risk_score = 0.0
                
                for vehicle in other_vehicles:
                    distance = vehicle['distance']
                    min_distance = min(min_distance, distance)
                    
                    # Simplified collision risk calculation
                    if distance < 50.0:  # Only consider nearby vehicles
                        risk_score = 1.0 / (1.0 + distance)
                        collision_risk_score = max(collision_risk_score, risk_score)
                
                collision_features = [
                    min_distance if min_distance != float('inf') else 1000.0,
                    min_time_to_collision if min_time_to_collision != float('inf') else 1000.0,
                    collision_risk_score,
                    0.0,  # closest_vehicle_angle
                    0.0,  # closest_vehicle_velocity_magnitude
                    0.0,  # closest_vehicle_relative_velocity
                    0.0,  # closest_vehicle_relative_position
                    np.sqrt(vx**2 + vy**2),  # ego_velocity_magnitude
                    np.sqrt(ax**2 + ay**2)   # ego_acceleration_magnitude
                ]
        
        # Combine base features + collision features (23 total)
        constraint_state = [
            x, y, vx, vy, ax, ay,  # 6 kinematic features
            lon_vel, lat_vel, lon_acc, lat_acc,  # 4 longitudinal/lateral features
            float(frame_num),  # Convert frame to float
            0.0, 0.0, 0.0,  # 3 additional features (placeholders)
        ] + collision_features  # 9 collision features
        
        return np.array(constraint_state, dtype=np.float32)

    def compute_reward(self, state, action, next_state, df=None, frame=None, goal_pos=None):
        """
        Compute reward based on Objective 1: Minimize Travel Time.
        
        Objective 1: Minimize T (travel time) while respecting constraints:
        - Hard convex constraints: dynamics, velocity/acceleration limits (MUST be satisfied)
        - Soft constraints: learned constraint model (should be satisfied)
        - Goal: reach destination in minimum time
        
        Args:
            state: Current state
            action: Selected action
            next_state: Next state
            df: DataFrame with other vehicles (for collision avoidance)
            frame: Current frame number
            goal_pos: Goal position (if None, use default)
        
        Returns:
            reward: Computed reward (negative if constraints violated)
            constraint_violated: Boolean indicating if hard constraints were violated
        """
        reward = 0.0
        constraint_violated = False
        
        # 1. PRIMARY OBJECTIVE: Minimize Travel Time
        current_pos = np.array([state[1], state[2]])  # x, y
        next_pos = np.array([next_state[1], next_state[2]])
        
        # Goal position - use provided goal or default
        if goal_pos is None:
            goal_pos = np.array([400.0, 300.0])  # Default goal position
        else:
            goal_pos = np.array(goal_pos)
            
        current_dist = np.linalg.norm(current_pos - goal_pos)
        next_dist = np.linalg.norm(next_pos - goal_pos)
        
        # Time minimization: reward for reducing distance to goal
        progress_reward = (current_dist - next_dist) * 20.0
        reward += progress_reward
        
        # 2. HARD CONVEX CONSTRAINTS: MUST be satisfied (no violations allowed)
        # Velocity constraint: ||v_t||_2 <= v_max
        velocity = np.array([next_state[3], next_state[4]])  # vx, vy
        speed = np.linalg.norm(velocity)
        v_max = 30.0  # Maximum velocity
        
        if speed > v_max:
            constraint_violated = True
            reward = -1000.0  # Severe penalty for hard constraint violation
            return reward, constraint_violated
        
        # Acceleration constraint: ||a_t||_2 <= a_max
        acceleration = np.array([next_state[5], next_state[6]])  # ax, ay
        accel_magnitude = np.linalg.norm(acceleration)
        a_max = 5.0  # Maximum acceleration
        
        if accel_magnitude > a_max:
            constraint_violated = True
            reward = -1000.0  # Severe penalty for hard constraint violation
            return reward, constraint_violated
        
        # 3. SOFT CONSTRAINTS: Learned constraint model (should be satisfied)
        if self.constraint_model is not None:
            with torch.no_grad():
                # Convert state to format expected by constraint model
                constraint_state = self.convert_state_for_constraint_model(next_state, df, frame)
                state_tensor = torch.FloatTensor(constraint_state).unsqueeze(0)
                constraint_likelihood = self.constraint_model(state_tensor).item()
                
                # Soft constraint satisfaction
                if constraint_likelihood >= 0.5:  # Satisfied
                    reward += 10.0  # Bonus for satisfying learned constraints
                elif constraint_likelihood < 0.3:  # Violated
                    reward -= 50.0  # Penalty for soft constraint violation
        
        # 4. COLLISION AVOIDANCE: Distance to other vehicles
        if df is not None and frame is not None:
            other_vehicles = get_nearby_vehicles(df, -1, frame, max_neighbors=5)
            if other_vehicles:
                min_distance = float('inf')
                for vehicle in other_vehicles:
                    distance = np.linalg.norm([vehicle['x'] - next_pos[0], vehicle['y'] - next_pos[1]])
                    min_distance = min(min_distance, distance)
                
                # Distance constraint: ||x_t - x_t^front||_2 >= d_min
                d_min = 10.0  # Minimum safe distance
                if min_distance < d_min:
                    constraint_violated = True
                    reward = -1000.0  # Severe penalty for collision risk
                    return reward, constraint_violated
                elif min_distance < d_min * 2:
                    reward -= 20.0  # Warning penalty for getting close
        
        # 5. GOAL REACHING: Terminate episode when goal reached
        if next_dist < 10.0:  # Close to goal
            # Episode will terminate, no additional reward needed
            pass
        
        # 6. TIME PENALTY: Encourage faster completion
        reward -= 1.0  # Time penalty per step
        
        return reward, constraint_violated
    
    def update_policy(self, batch_size=32):
        """
        Update RL policy network using experience replay.
        NOTE: This trains the RL policy, NOT the constraint model.
        The constraint model is only used for inference/evaluation.
        
        Args:
            batch_size: Size of training batch
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Get constraint masks (using pre-trained constraint model for inference only)
        current_masks = self.policy_net.get_constraint_mask(states, self.action_space)
        next_masks = self.policy_net.get_constraint_mask(next_states, self.action_space)
        
        # Get current and next action probabilities and values
        current_logits, current_probs, current_values = self.policy_net(states, current_masks)
        next_logits, next_probs, next_values = self.policy_net(next_states, next_masks)
        
        # Compute target values
        target_values = rewards + self.gamma * next_values.squeeze() * (~dones).float()
        
        # Policy loss (cross-entropy with constraint masking)
        action_probs = current_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -torch.log(action_probs + 1e-8).mean()
        
        # Value loss (MSE)
        value_loss = F.mse_loss(current_values.squeeze(), target_values.detach())
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Backward pass (only updates RL policy network, not constraint model)
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Record statistics
        self.training_stats['policy_losses'].append(policy_loss.item())
        self.training_stats['value_losses'].append(value_loss.item())
        
        return policy_loss.item(), value_loss.item()
    
    def train_episode(self, initial_state, df, max_steps=100, goal_pos=None):
        """
        Train for one episode with travel time optimization metrics tracking.
        
        Args:
            initial_state: Starting state
            df: DataFrame with other vehicles
            max_steps: Maximum steps per episode
            goal_pos: Goal position (if None, use default)
        
        Returns:
            episode_reward: Total reward for the episode
            constraint_violations: Number of constraint violations
        """
        state = initial_state.copy()
        episode_reward = 0.0
        constraint_violations = 0
        frame = 0
        action_accepted = True  # Track if action was accepted
        
        # Travel time optimization tracking
        episode_velocity_violations = 0
        episode_acceleration_violations = 0
        episode_collision_risks = 0
        episode_constraint_satisfaction = []
        
        # Trajectory data collection
        trajectory_states = []
        trajectory_rewards = []
        trajectory_times = []
        
        # Set goal position
        if goal_pos is None:
            goal_pos = np.array([400.0, 300.0])  # Default goal position
        else:
            goal_pos = np.array(goal_pos)
            
        initial_distance = np.linalg.norm([state[1] - goal_pos[0], state[2] - goal_pos[1]])
        
        for step in range(max_steps):
            # Select action
            action_idx, action_probs = self.select_action(state, training=True)
            action = self.action_space[action_idx]
            
            # Apply action to get next state
            next_state = self.policy_net.apply_action_to_state(
                torch.FloatTensor(state).unsqueeze(0), action
            ).squeeze().numpy()
            
            # Track hard convex constraints violations
            # Velocity constraint: ||v_t||_2 <= v_max
            velocity = np.array([next_state[3], next_state[4]])
            speed = np.linalg.norm(velocity)
            v_max = 30.0
            if speed > v_max:
                episode_velocity_violations += 1
            
            # Acceleration constraint: ||a_t||_2 <= a_max
            acceleration = np.array([next_state[5], next_state[6]])
            accel_magnitude = np.linalg.norm(acceleration)
            a_max = 5.0
            if accel_magnitude > a_max:
                episode_acceleration_violations += 1
            
            # Track soft constraints (using pre-trained constraint model for inference only)
            if self.constraint_model is not None:
                with torch.no_grad():  # No gradients - constraint model is not being trained
                    # Convert state to format expected by constraint model
                    constraint_state = self.convert_state_for_constraint_model(next_state, df, frame)
                    state_tensor = torch.FloatTensor(constraint_state).unsqueeze(0)
                    constraint_likelihood = self.constraint_model(state_tensor).item()
                    episode_constraint_satisfaction.append(constraint_likelihood)
                    
                    if constraint_likelihood < 0.3:
                        constraint_violations += 1
            
            # Track collision risks
            if df is not None and frame is not None:
                other_vehicles = get_nearby_vehicles(df, -1, frame, max_neighbors=5)
                if other_vehicles:
                    min_distance = float('inf')
                    for vehicle in other_vehicles:
                        distance = np.linalg.norm([vehicle['x'] - next_state[1], vehicle['y'] - next_state[2]])
                        min_distance = min(min_distance, distance)
                    
                    if min_distance < 10.0:  # d_min
                        episode_collision_risks += 1
            
            # Compute reward with goal position
            reward, constraint_violated = self.compute_reward(state, action, next_state, df, frame, goal_pos)
            
            # If hard constraints are violated, reject this action and try again
            if constraint_violated:
                # Try alternative actions that might satisfy constraints
                alternative_actions = [i for i in range(len(self.action_space)) if i != action_idx]
                action_accepted = False
                
                for alt_action_idx in alternative_actions:
                    alt_action = self.action_space[alt_action_idx]
                    alt_next_state = self.policy_net.apply_action_to_state(
                        torch.FloatTensor(state).unsqueeze(0), alt_action
                    ).squeeze().numpy()
                    
                    alt_reward, alt_constraint_violated = self.compute_reward(
                        state, alt_action, alt_next_state, df, frame, goal_pos
                    )
                    
                    if not alt_constraint_violated:
                        # Use this alternative action instead
                        action_idx = alt_action_idx
                        action = alt_action
                        next_state = alt_next_state
                        reward = alt_reward
                        action_accepted = True
                        break
                
                if not action_accepted:
                    # If no alternative action satisfies constraints, use original but with severe penalty
                    constraint_violations += 1
                    reward = -1000.0  # Severe penalty for unavoidable constraint violation
            else:
                action_accepted = True
            
            episode_reward += reward
            
            # Collect trajectory data
            trajectory_states.append({
                'x': state[1],
                'y': state[2], 
                'vx': state[3],
                'vy': state[4],
                'ax': state[5],
                'ay': state[6],
                't': frame * 0.4  # Time in seconds (assuming 0.4s timestep)
            })
            trajectory_rewards.append(reward)
            trajectory_times.append(frame * 0.4)
            
            # Check if episode is done
            done = False
            current_distance = np.linalg.norm([next_state[1] - goal_pos[0], next_state[2] - goal_pos[1]])
            
            if step == max_steps - 1:
                done = True
            elif current_distance < 10.0:  # Reached goal
                done = True
            elif np.linalg.norm([next_state[1], next_state[2]]) > 1000:  # Out of bounds
                done = True
                reward -= 100.0  # Penalty for going out of bounds
            
            # Store experience - be extremely lenient during training
            # Allow almost all experiences to help the agent learn, even with violations
            if not constraint_violated or random.random() < 0.9:  # 90% chance to store violations
                self.replay_buffer.append((state, action_idx, reward, next_state, done))
            else:
                # Skip this experience - don't learn from constraint violations
                # This forces the agent to learn only from valid trajectories
                pass
            
            # Update state
            state = next_state
            frame += 1
            
            if done:
                break
        
        # Update policy (only if we have enough constraint-satisfying experiences)
        if len(self.replay_buffer) >= 32:
            self.update_policy(batch_size=32)
        
        # Early termination if too many constraint violations
        # This prevents learning from episodes with frequent violations
        if constraint_violations > max_steps * 0.7:  # More than 70% violations (very lenient)
            print(f"Episode terminated early due to excessive constraint violations: {constraint_violations}/{max_steps}")
            episode_reward = -300.0  # Reduced penalty for episode with too many violations
        
        # Record travel time optimization statistics
        travel_time = frame  # Number of steps taken
        final_distance = np.linalg.norm([state[1] - goal_pos[0], state[2] - goal_pos[1]])
        
        self.training_stats['travel_times'].append(travel_time)
        self.training_stats['goal_distances'].append(final_distance)
        self.training_stats['velocity_violations'].append(episode_velocity_violations)
        self.training_stats['acceleration_violations'].append(episode_acceleration_violations)
        self.training_stats['collision_risks'].append(episode_collision_risks)
        if episode_constraint_satisfaction:
            self.training_stats['constraint_satisfaction'].append(np.mean(episode_constraint_satisfaction))
        
        # Record trajectory data
        trajectory_data = {
            'episode': len(self.training_stats['episode_rewards']),
            'states': trajectory_states,
            'rewards': trajectory_rewards,
            'times': trajectory_times,
            'total_reward': episode_reward,
            'travel_time': travel_time,
            'final_distance': final_distance,
            'goal_pos': goal_pos.tolist()
        }
        self.training_stats['trajectory_data'].append(trajectory_data)
        
        return episode_reward, constraint_violations
    
    def save_model(self, filepath):
        """Save the trained RL model."""
        model_data = {
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'action_space': self.action_space,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'learning_rate': self.learning_rate
            }
        }
        torch.save(model_data, filepath)
        print(f"RL model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained RL model."""
        if os.path.exists(filepath):
            model_data = torch.load(filepath)
            self.policy_net.load_state_dict(model_data['policy_state_dict'])
            self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
            self.training_stats = model_data['training_stats']
            self.action_space = model_data['action_space']
            print(f"RL model loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found")

    def save_trajectory_data(self, output_dir="./trajectory_data"):
        """
        Save trajectory data to CSV files for analysis and plotting.
        
        Args:
            output_dir: Directory to save trajectory data
        """
        if not self.training_stats['trajectory_data']:
            print("No trajectory data to save")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all trajectory data
        all_trajectories = []
        for traj_data in self.training_stats['trajectory_data']:
            episode = traj_data['episode']
            states = traj_data['states']
            rewards = traj_data['rewards']
            times = traj_data['times']
            
            for i, (state, reward, time) in enumerate(zip(states, rewards, times)):
                all_trajectories.append({
                    'episode': episode,
                    'step': i,
                    'time': time,
                    'x': state['x'],
                    'y': state['y'],
                    'vx': state['vx'],
                    'vy': state['vy'],
                    'ax': state['ax'],
                    'ay': state['ay'],
                    'reward': reward,
                    'total_reward': traj_data['total_reward'],
                    'travel_time': traj_data['travel_time'],
                    'final_distance': traj_data['final_distance']
                })
        
        # Save to CSV
        df = pd.DataFrame(all_trajectories)
        csv_path = os.path.join(output_dir, "rl_trajectories.csv")
        df.to_csv(csv_path, index=False)
        print(f"Trajectory data saved to {csv_path}")
        
        # Save episode summary
        episode_summary = []
        for traj_data in self.training_stats['trajectory_data']:
            episode_summary.append({
                'episode': traj_data['episode'],
                'total_reward': traj_data['total_reward'],
                'travel_time': traj_data['travel_time'],
                'final_distance': traj_data['final_distance'],
                'num_steps': len(traj_data['states']),
                'goal_x': traj_data['goal_pos'][0],
                'goal_y': traj_data['goal_pos'][1]
            })
        
        df_summary = pd.DataFrame(episode_summary)
        summary_path = os.path.join(output_dir, "rl_episode_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"Episode summary saved to {summary_path}")
        
        return csv_path, summary_path


def extract_realistic_goals(df, num_goals=10):
    """
    Extract realistic goal positions from trajectory data.
    
    Args:
        df: DataFrame with trajectory data
        num_goals: Number of goals to extract
    
    Returns:
        goals: List of goal positions
    """
    goals = []
    
    # Get unique tracks
    unique_tracks = df['trackId'].unique()
    
    for track_id in unique_tracks[:num_goals]:
        track_data = df[df['trackId'] == track_id]
        if len(track_data) > 10:  # Only use tracks with sufficient data
            # Use the final position as a goal
            final_x = track_data['xCenter'].iloc[-1]
            final_y = track_data['yCenter'].iloc[-1]
            goals.append([final_x, final_y])
    
    # If we don't have enough goals, add some reasonable defaults
    while len(goals) < num_goals:
        goals.append([400.0, 300.0])  # Default goal
    
    return goals

def create_rl_training_data(csv_file, constraint_model_path=None, num_episodes=1000):
    """
    Train RL agent using pre-trained constraint model for inference.
    
    NOTE: This function trains the RL policy network, NOT the constraint model.
    The constraint model is loaded from the provided path and used only for
    inference/evaluation during RL training.
    
    Args:
        csv_file: Path to CSV file with trajectory data
        constraint_model_path: Path to pre-trained constraint model (for inference only)
        num_episodes: Number of episodes to generate
    
    Returns:
        rl_agent: Trained RL agent
    """
    # Load data
    df = pd.read_csv(csv_file)
    
    # Extract realistic goals from trajectory data
    realistic_goals = extract_realistic_goals(df, num_goals=20)
    
    # Initialize RL agent
    rl_agent = ConstraintGuidedRL(
        state_dim=23,
        action_dim=9,  # 9 discrete actions
        constraint_model_path=constraint_model_path
    )
    
    print(f"Training RL agent for {num_episodes} episodes...")
    print("NOTE: Training RL policy network using pre-trained constraint model for inference only.")
    print("The constraint model itself is NOT being retrained.")
    
    for episode in range(num_episodes):
        # Sample random initial state from data
        random_track = df['trackId'].sample(1).iloc[0]
        track_data = df[df['trackId'] == random_track]
        
        if len(track_data) < 10:
            continue
        
        # Get initial state (first few frames)
        initial_frame = track_data['frame'].iloc[0]
        initial_state = [
            random_track,  # track_id
            track_data['xCenter'].iloc[0],  # x
            track_data['yCenter'].iloc[0],  # y
            track_data['xVelocity'].iloc[0],  # vx
            track_data['yVelocity'].iloc[0],  # vy
            track_data['xAcceleration'].iloc[0],  # ax
            track_data['yAcceleration'].iloc[0],  # ay
            track_data['lonVelocity'].iloc[0],  # lon_vel
            track_data['latVelocity'].iloc[0],  # lat_vel
            track_data['lonAcceleration'].iloc[0],  # lon_acc
            track_data['latAcceleration'].iloc[0],  # lat_acc
            initial_frame,  # frame
            # Add relative features (simplified) - need 11 features to make total 23
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 11 placeholder relative features
        ]
        
        # Select a realistic goal for this episode
        episode_goal = random.choice(realistic_goals)
        
        # Train episode with specific goal
        episode_reward, constraint_violations = rl_agent.train_episode(
            initial_state, df, max_steps=50, goal_pos=episode_goal
        )
        
        # Record statistics
        rl_agent.training_stats['episode_rewards'].append(episode_reward)
        rl_agent.training_stats['constraint_violations'].append(constraint_violations)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rl_agent.training_stats['episode_rewards'][-100:])
            avg_violations = np.mean(rl_agent.training_stats['constraint_violations'][-100:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, Avg Violations = {avg_violations:.2f}")
    
    return rl_agent


def generate_rl_trajectories(rl_agent, start_state, df, max_steps=100):
    """
    Generate trajectories using the trained RL agent.
    
    Args:
        rl_agent: Trained RL agent
        start_state: Initial state
        df: DataFrame with other vehicles
        max_steps: Maximum trajectory length
    
    Returns:
        trajectory: List of states
        actions: List of actions
        rewards: List of rewards
    """
    trajectory = [start_state]
    actions = []
    rewards = []
    
    state = start_state.copy()
    frame = 0
    
    for step in range(max_steps):
        # Select action (no exploration during generation)
        action_idx, action_probs = rl_agent.select_action(state, training=False)
        action = rl_agent.action_space[action_idx]
        
        # Apply action
        next_state = rl_agent.policy_net.apply_action_to_state(
            torch.FloatTensor(state).unsqueeze(0), action
        ).squeeze().numpy()
        
        # Compute reward
        reward, constraint_violated = rl_agent.compute_reward(state, action, next_state, df, frame)
        
        # If hard constraints are violated, try alternative actions
        if constraint_violated:
            # Try alternative actions that might satisfy constraints
            alternative_actions = [i for i in range(len(rl_agent.action_space)) if i != action_idx]
            action_accepted = False
            
            for alt_action_idx in alternative_actions:
                alt_action = rl_agent.action_space[alt_action_idx]
                alt_next_state = rl_agent.policy_net.apply_action_to_state(
                    torch.FloatTensor(state).unsqueeze(0), alt_action
                ).squeeze().numpy()
                
                alt_reward, alt_constraint_violated = rl_agent.compute_reward(
                    state, alt_action, alt_next_state, df, frame
                )
                
                if not alt_constraint_violated:
                    # Use this alternative action instead
                    action_idx = alt_action_idx
                    action = alt_action
                    next_state = alt_next_state
                    reward = alt_reward
                    action_accepted = True
                    break
            
            if not action_accepted:
                # If no alternative action satisfies constraints, use original but with severe penalty
                reward = -1000.0  # Severe penalty for unavoidable constraint violation
        
        # Store results
        trajectory.append(next_state)
        actions.append(action)
        rewards.append(reward)
        
        # Update state
        state = next_state
        frame += 1
        
        # Check termination
        if np.linalg.norm([next_state[1], next_state[2]]) > 1000:
            break
    
    return trajectory, actions, rewards


if __name__ == "__main__":
    # Example usage
    csv_file = "./inD/00_tracks.csv"
    constraint_model_path = "./model_checkpoint/model_00_tracks_f357680b.pth"
    
    # Train RL agent
    rl_agent = create_rl_training_data(
        csv_file=csv_file,
        constraint_model_path=constraint_model_path,
        num_episodes=500
    )
    
    # Save trained agent
    rl_agent.save_model("./rl_agent_trained.pth")
    
    # Generate sample trajectory
    sample_start_state = [0, 100.0, 100.0, 10.0, 5.0, 0.0, 0.0, 10.0, 5.0, 0.0, 0.0, 0] + [0.0] * 11
    df = pd.read_csv(csv_file)
    
    trajectory, actions, rewards = generate_rl_trajectories(
        rl_agent, sample_start_state, df, max_steps=50
    )
    
    print(f"Generated trajectory with {len(trajectory)} states")
    print(f"Total reward: {sum(rewards):.2f}")
    print(f"Final position: ({trajectory[-1][1]:.2f}, {trajectory[-1][2]:.2f})") 