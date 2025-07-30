#!/usr/bin/env python3
"""
RL with Constraints - Convex Optimization Version
This version uses convex optimization techniques for constraint satisfaction.
"""

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
import cvxpy as cp

# Import your existing constraint model
from constraint_model_utils import TransitionPredictionNN, load_model, get_nearby_vehicles, create_collision_avoidance_features

class ConvexConstraintAwarePolicyNetwork(nn.Module):
    """
    Policy network that uses convex optimization for constraint satisfaction.
    """
    def __init__(self, state_dim=23, action_dim=4, hidden_dim=128, constraint_model=None):
        super(ConvexConstraintAwarePolicyNetwork, self).__init__()
        
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
        
        # Value network for critic
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
        Forward pass with convex optimization for constraint satisfaction.
        """
        # Get raw action logits
        action_logits = self.policy_net(state)
        value = self.value_net(state)
        
        # Apply convex optimization for constraint satisfaction
        if constraint_mask is not None:
            action_logits = self._apply_convex_constraints(action_logits, constraint_mask)
        
        # Convert to probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_logits, action_probs, value
    
    def _apply_convex_constraints(self, action_logits, constraint_mask):
        """
        Apply convex optimization to satisfy constraints.
        """
        try:
            # Convert to numpy for convex optimization
            logits_np = action_logits.detach().numpy()
            mask_np = constraint_mask.detach().numpy()
            
            # Define convex optimization problem
            action_var = cp.Variable(logits_np.shape[1])
            objective = cp.Minimize(cp.sum_squares(action_var - logits_np.flatten()))
            constraints = [action_var >= 0, cp.sum(action_var) == 1]
            
            # Add constraint mask constraints
            for i in range(len(mask_np.flatten())):
                if mask_np.flatten()[i] == 0:
                    constraints.append(action_var[i] == 0)
            
            # Solve the convex optimization problem
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimized_actions = action_var.value
                return torch.tensor(optimized_actions.reshape(logits_np.shape), dtype=torch.float32)
            else:
                # Fallback to original logits if optimization fails
                return action_logits
                
        except Exception as e:
            print(f"Convex optimization failed: {e}, using fallback")
            return action_logits

class ConvexConstraintGuidedRL:
    """
    RL agent that uses convex optimization for constraint satisfaction.
    """
    def __init__(self, state_dim=23, action_dim=9, constraint_model_path=None, 
                 learning_rate=0.001, gamma=0.99, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize policy network
        self.policy_network = ConvexConstraintAwarePolicyNetwork(
            state_dim=state_dim, 
            action_dim=action_dim,
            constraint_model=self._load_constraint_model(constraint_model_path)
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
        # Training statistics
        self.training_rewards = []
        self.constraint_violations = []
        
        print(f"✓ Convex Constraint-Guided RL initialized")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Device: {self.device}")
    
    def _load_constraint_model(self, constraint_model_path):
        """Load the pre-trained constraint model."""
        if constraint_model_path and os.path.exists(constraint_model_path):
            try:
                constraint_model = load_model(constraint_model_path)
                print(f"✓ Constraint model loaded from {constraint_model_path}")
                return constraint_model
            except Exception as e:
                print(f"⚠️  Failed to load constraint model: {e}")
                return None
        else:
            print("⚠️  No constraint model path provided")
            return None
    
    def select_action(self, state, training=True):
        """
        Select action using convex optimization for constraint satisfaction.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get constraint mask
        constraint_mask = self._get_constraint_mask(state_tensor)
        
        # Get action probabilities with convex optimization
        with torch.no_grad():
            action_logits, action_probs, value = self.policy_network(state_tensor, constraint_mask)
        
        if training and random.random() < self.epsilon:
            # Epsilon-greedy exploration
            action = random.randint(0, self.action_dim - 1)
        else:
            # Select action based on probabilities
            action_probs_np = action_probs.cpu().numpy().flatten()
            action = np.random.choice(len(action_probs_np), p=action_probs_np)
        
        return action, action_probs.cpu().numpy().flatten()
    
    def _get_constraint_mask(self, state_tensor):
        """Generate constraint mask using the learned constraint model."""
        if self.policy_network.constraint_model is None:
            return torch.ones(state_tensor.shape[0], self.action_dim).to(self.device)
        
        try:
            # Get constraint satisfaction probability
            constraint_prob = self.policy_network.constraint_model(state_tensor)
            
            # Create mask based on constraint satisfaction
            mask = torch.ones(state_tensor.shape[0], self.action_dim).to(self.device)
            
            # Apply constraint mask based on satisfaction probability
            if constraint_prob.item() < 0.5:
                # If constraint is likely to be violated, mask some actions
                mask = mask * 0.8  # Reduce probability of all actions
            
            return mask
            
        except Exception as e:
            print(f"⚠️  Error generating constraint mask: {e}")
            return torch.ones(state_tensor.shape[0], self.action_dim).to(self.device)
    
    def _apply_action_to_state(self, state, action):
        """Apply action to state using convex optimization."""
        dt = 0.4  # Time step
        
        # Convert action to velocity changes
        action_scale = 2.0
        dvx = (action % 3 - 1) * action_scale  # -2, 0, 2
        dvy = (action // 3 - 1) * action_scale  # -2, 0, 2
        
        # Apply convex optimization for smooth transitions
        try:
            # Define convex optimization for velocity changes
            dv = cp.Variable(2)
            objective = cp.Minimize(cp.sum_squares(dv - np.array([dvx, dvy])))
            constraints = [
                cp.norm(dv, 2) <= 3.0,  # Maximum velocity change
                dv >= -2.0,
                dv <= 2.0
            ]
            
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimized_dv = dv.value
                dvx, dvy = optimized_dv[0], optimized_dv[1]
            else:
                # Fallback to original values
                pass
                
        except Exception as e:
            print(f"Convex optimization failed: {e}")
        
        # Update velocities
        new_vx = state[3] + dvx
        new_vy = state[4] + dvy
        
        # Apply velocity constraints
        max_velocity = 30.0
        velocity_magnitude = np.sqrt(new_vx**2 + new_vy**2)
        if velocity_magnitude > max_velocity:
            scale = max_velocity / velocity_magnitude
            new_vx *= scale
            new_vy *= scale
        
        # Update positions
        new_x = state[1] + new_vx * dt
        new_y = state[2] + new_vy * dt
        
        # Calculate accelerations
        ax = dvx / dt
        ay = dvy / dt
        
        # Apply acceleration constraints
        max_acceleration = 5.0
        acceleration_magnitude = np.sqrt(ax**2 + ay**2)
        if acceleration_magnitude > max_acceleration:
            scale = max_acceleration / acceleration_magnitude
            ax *= scale
            ay *= scale
        
        # Create new state
        new_state = list(state)
        new_state[1] = new_x  # x
        new_state[2] = new_y  # y
        new_state[3] = new_vx  # vx
        new_state[4] = new_vy  # vy
        new_state[5] = ax  # ax
        new_state[6] = ay  # ay
        
        return new_state
    
    def compute_reward(self, state, action, next_state, df=None, frame=None, goal_pos=None):
        """Compute reward with convex optimization considerations."""
        reward = 0.0
        constraint_violated = False
        
        # Extract state components
        x, y = next_state[1], next_state[2]
        vx, vy = next_state[3], next_state[4]
        ax, ay = next_state[5], next_state[6]
        
        # Calculate magnitudes
        speed = np.sqrt(vx**2 + vy**2)
        accel_magnitude = np.sqrt(ax**2 + ay**2)
        
        # 1. Progress towards goal
        if goal_pos is not None:
            current_dist = np.sqrt((x - goal_pos[0])**2 + (y - goal_pos[1])**2)
            reward += (100.0 - current_dist) * 0.1
        else:
            # Reward for forward progress
            reward += x * 0.1
        
        # 2. Hard constraints (must be satisfied)
        v_max, a_max, d_min = 30.0, 5.0, 10.0
        
        if speed > v_max or accel_magnitude > a_max:
            constraint_violated = True
            reward = -1000.0
            return reward, constraint_violated
        
        # 3. Soft constraints (learned from data)
        if self.policy_network.constraint_model is not None:
            try:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                constraint_likelihood = self.policy_network.constraint_model(state_tensor).item()
                
                if constraint_likelihood >= 0.5:
                    reward += 10.0  # Bonus for satisfying learned constraints
                elif constraint_likelihood < 0.3:
                    reward -= 50.0  # Penalty for violating learned constraints
            except Exception as e:
                print(f"⚠️  Error computing constraint reward: {e}")
        
        # 4. Convex optimization bonus
        try:
            # Bonus for smooth, convex-optimized trajectories
            velocity_change = np.sqrt((vx - state[3])**2 + (vy - state[4])**2)
            if velocity_change < 2.0:  # Smooth transition
                reward += 5.0
        except:
            pass
        
        # 5. Time penalty
        reward -= 1.0
        
        return reward, constraint_violated
    
    def update_policy(self, batch_size=32):
        """Update policy using convex optimization techniques."""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Get current Q-values
        current_q_values = self.policy_network(states)[0]
        current_q = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Get next Q-values
        next_q_values = self.policy_network(next_states)[0]
        next_q = next_q_values.max(1)[0].detach()
        
        # Compute target Q-values
        target_q = rewards + (self.gamma * next_q * ~dones)
        
        # Compute loss with convex regularization
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Add convex regularization term
        l2_reg = torch.tensor(0.0).to(self.device)
        for param in self.policy_network.parameters():
            l2_reg += torch.norm(param)
        loss += 0.001 * l2_reg
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self, initial_state, df, max_steps=100, goal_pos=None):
        """Train for one episode using convex optimization."""
        state = initial_state.copy()
        episode_reward = 0
        episode_violations = 0
        trajectory = [state]
        
        for step in range(max_steps):
            # Select action
            action, action_probs = self.select_action(state, training=True)
            
            # Apply action
            next_state = self._apply_action_to_state(state, action)
            
            # Compute reward
            reward, constraint_violated = self.compute_reward(state, action, next_state, df, step, goal_pos)
            
            if constraint_violated:
                episode_violations += 1
                if episode_violations > 50:  # Early termination
                    break
            
            # Store experience
            done = (step == max_steps - 1)
            self.memory.append((state, action, reward, next_state, done))
            
            # Update state
            state = next_state
            trajectory.append(state)
            episode_reward += reward
            
            # Update policy
            if len(self.memory) >= 32:
                loss = self.update_policy()
        
        return episode_reward, episode_violations, trajectory
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_rewards': self.training_rewards,
            'constraint_violations': self.constraint_violations
        }, filepath)
        print(f"✓ Convex RL model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_rewards = checkpoint.get('training_rewards', [])
        self.constraint_violations = checkpoint.get('constraint_violations', [])
        print(f"✓ Convex RL model loaded from {filepath}")
    
    def save_trajectory_data(self, output_dir="./trajectory_data"):
        """Save trajectory data to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training history
        training_data = {
            'episode': range(1, len(self.training_rewards) + 1),
            'reward': self.training_rewards,
            'constraint_violations': self.constraint_violations
        }
        
        df_training = pd.DataFrame(training_data)
        training_path = os.path.join(output_dir, "convex_rl_training_history.csv")
        df_training.to_csv(training_path, index=False)
        
        # Save episode summary
        summary_data = {
            'metric': ['Total Episodes', 'Average Reward', 'Average Violations', 'Best Reward', 'Worst Reward'],
            'value': [
                len(self.training_rewards),
                np.mean(self.training_rewards) if self.training_rewards else 0,
                np.mean(self.constraint_violations) if self.constraint_violations else 0,
                np.max(self.training_rewards) if self.training_rewards else 0,
                np.min(self.training_rewards) if self.training_rewards else 0
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "convex_rl_episode_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        
        print(f"✓ Convex RL trajectory data saved to {output_dir}")
        return training_path, summary_path

def create_convex_rl_training_data(csv_file, constraint_model_path=None, num_episodes=1000):
    """Create and train a convex constraint-guided RL agent."""
    print(f"=== Creating Convex Constraint-Guided RL Training Data ===")
    print(f"CSV file: {csv_file}")
    print(f"Constraint model: {constraint_model_path}")
    print(f"Episodes: {num_episodes}")
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} samples from {df['trackId'].nunique()} tracks")
    
    # Initialize RL agent
    rl_agent = ConvexConstraintGuidedRL(
        constraint_model_path=constraint_model_path,
        num_episodes=num_episodes
    )
    
    # Training loop
    print(f"\nTraining convex RL agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Sample initial state from data
        random_track = df['trackId'].sample(1).iloc[0]
        track_data = df[df['trackId'] == random_track]
        
        if len(track_data) < 10:
            continue
        
        # Create initial state
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
            track_data['frame'].iloc[0],  # frame
            # Add relative features (simplified)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        
        # Train episode
        episode_reward, episode_violations, trajectory = rl_agent.train_episode(
            initial_state, df, max_steps=100
        )
        
        # Store statistics
        rl_agent.training_rewards.append(episode_reward)
        rl_agent.constraint_violations.append(episode_violations)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rl_agent.training_rewards[-100:])
            avg_violations = np.mean(rl_agent.constraint_violations[-100:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, Avg Violations = {avg_violations:.2f}")
    
    print(f"✓ Convex RL training completed")
    return rl_agent

def generate_convex_rl_trajectories(rl_agent, start_state, df, max_steps=100):
    """Generate trajectories using the trained convex RL agent."""
    state = start_state.copy()
    trajectory = [state]
    actions = []
    rewards = []
    
    for step in range(max_steps):
        # Select action
        action, action_probs = rl_agent.select_action(state, training=False)
        
        # Apply action
        next_state = rl_agent._apply_action_to_state(state, action)
        
        # Compute reward
        reward, constraint_violated = rl_agent.compute_reward(state, action, next_state, df, step)
        
        # Store step
        trajectory.append(next_state)
        actions.append(action)
        rewards.append(reward)
        
        # Update state
        state = next_state
        
        # Early termination if too many violations
        if constraint_violated and len([r for r in rewards if r < -100]) > 10:
            break
    
    return trajectory, actions, rewards

if __name__ == "__main__":
    # Example usage
    csv_file = "./inD/00_tracks.csv"
    constraint_model_path = "./model_checkpoint/model_00_tracks_f357680b.pth"
    
    # Create and train convex RL agent
    rl_agent = create_convex_rl_training_data(
        csv_file=csv_file,
        constraint_model_path=constraint_model_path,
        num_episodes=500
    )
    
    # Save trained agent
    rl_agent.save_model("./rl_agent_trained_convex.pth")
    
    # Save trajectory data
    rl_agent.save_trajectory_data("./trajectory_data")
    
    print("✓ Convex RL with constraints training completed!") 