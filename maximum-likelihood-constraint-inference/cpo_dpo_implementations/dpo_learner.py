#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) implementation for constraint learning.
Based on the paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PreferencePolicyNetwork(nn.Module):
    """Policy network for DPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(PreferencePolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        logstd = self.fc_logstd(x)
        return mean, logstd

class ReferencePolicyNetwork(nn.Module):
    """Reference policy network for DPO (usually a pre-trained policy)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ReferencePolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        logstd = self.fc_logstd(x)
        return mean, logstd

class DPOLearner:
    """
    Direct Preference Optimization (DPO) implementation.
    
    This implementation learns from preference data (preferred vs less preferred trajectories)
    without requiring explicit reward functions.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 beta: float = 0.1,
                 lr: float = 3e-4,
                 device: str = 'cpu'):
        """
        Initialize DPO learner.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            beta: Temperature parameter for DPO (controls how much to deviate from reference policy)
            lr: Learning rate
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta
        self.device = device
        
        # Networks
        self.policy = PreferencePolicyNetwork(state_dim, action_dim).to(device)
        self.reference_policy = ReferencePolicyNetwork(state_dim, action_dim).to(device)
        
        # Initialize reference policy with same weights as policy
        self.reference_policy.load_state_dict(self.policy.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage for preference data
        self.preference_pairs = []
        
    def get_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Get action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, logstd = self.policy(state_tensor)
        
        # Sample action
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze(0).detach().cpu().numpy(), log_prob.item()
    
    def compute_log_probs(self, states: torch.Tensor, actions: torch.Tensor, policy: nn.Module) -> torch.Tensor:
        """Compute log probabilities for given state-action pairs."""
        mean, logstd = policy(states)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return log_probs
    
    def compute_preference_loss(self, 
                              preferred_states: torch.Tensor,
                              preferred_actions: torch.Tensor,
                              less_preferred_states: torch.Tensor,
                              less_preferred_actions: torch.Tensor) -> torch.Tensor:
        """
        Compute DPO preference loss.
        
        Args:
            preferred_states: States from preferred trajectories
            preferred_actions: Actions from preferred trajectories
            less_preferred_states: States from less preferred trajectories
            less_preferred_actions: Actions from less preferred trajectories
        """
        
        # Compute log probabilities for current policy
        preferred_log_probs = self.compute_log_probs(preferred_states, preferred_actions, self.policy)
        less_preferred_log_probs = self.compute_log_probs(less_preferred_states, less_preferred_actions, self.policy)
        
        # Compute log probabilities for reference policy
        preferred_ref_log_probs = self.compute_log_probs(preferred_states, preferred_actions, self.reference_policy)
        less_preferred_ref_log_probs = self.compute_log_probs(less_preferred_states, less_preferred_actions, self.reference_policy)
        
        # Compute log ratios
        preferred_log_ratio = preferred_log_probs - preferred_ref_log_probs
        less_preferred_log_ratio = less_preferred_log_probs - less_preferred_ref_log_probs
        
        # Compute DPO loss
        # L_DPO = -log(σ(β * (r_θ(y_w) - r_θ(y_l))))
        # where r_θ(y) = log π_θ(y) - log π_ref(y)
        log_odds = self.beta * (preferred_log_ratio - less_preferred_log_ratio)
        loss = -torch.log(torch.sigmoid(log_odds)).mean()
        
        return loss
    
    def add_preference_pair(self, 
                          preferred_trajectory: List[Tuple[np.ndarray, np.ndarray]],
                          less_preferred_trajectory: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Add a preference pair to the training data.
        
        Args:
            preferred_trajectory: List of (state, action) tuples from preferred trajectory
            less_preferred_trajectory: List of (state, action) tuples from less preferred trajectory
        """
        self.preference_pairs.append((preferred_trajectory, less_preferred_trajectory))
    
    def create_constraint_preference_data(self, 
                                        demonstrations: List[List[Tuple[np.ndarray, np.ndarray]]],
                                        constraint_functions: List[callable],
                                        constraint_limits: List[float]) -> List[Tuple[List, List]]:
        """
        Create preference data from demonstrations based on constraint satisfaction.
        
        Args:
            demonstrations: List of trajectories, each trajectory is a list of (state, action) tuples
            constraint_functions: List of constraint functions
            constraint_limits: List of constraint limits
            
        Returns:
            List of preference pairs (preferred_traj, less_preferred_traj)
        """
        preference_pairs = []
        
        for i, traj1 in enumerate(demonstrations):
            for j, traj2 in enumerate(demonstrations[i+1:], i+1):
                # Compute constraint violations for both trajectories
                violations1 = self.compute_trajectory_constraint_violations(traj1, constraint_functions, constraint_limits)
                violations2 = self.compute_trajectory_constraint_violations(traj2, constraint_functions, constraint_limits)
                
                # Determine which trajectory is preferred based on constraint satisfaction
                total_violation1 = np.sum(np.maximum(violations1, 0))  # Only positive violations
                total_violation2 = np.sum(np.maximum(violations2, 0))
                
                if total_violation1 < total_violation2:
                    # traj1 is preferred (fewer constraint violations)
                    preference_pairs.append((traj1, traj2))
                elif total_violation2 < total_violation1:
                    # traj2 is preferred (fewer constraint violations)
                    preference_pairs.append((traj2, traj1))
                # If violations are equal, skip this pair
        
        return preference_pairs
    
    def compute_trajectory_constraint_violations(self, 
                                              trajectory: List[Tuple[np.ndarray, np.ndarray]],
                                              constraint_functions: List[callable],
                                              constraint_limits: List[float]) -> np.ndarray:
        """Compute constraint violations for a trajectory."""
        violations = []
        for state, action in trajectory:
            traj_violations = []
            for constraint_fn, limit in zip(constraint_functions, constraint_limits):
                value = constraint_fn(state, action)
                violation = value - limit  # Positive means violation
                traj_violations.append(violation)
            violations.append(traj_violations)
        return np.array(violations)
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """Perform one training step with preference data."""
        if len(self.preference_pairs) < batch_size:
            logger.warning(f"Not enough preference pairs ({len(self.preference_pairs)}) for batch size {batch_size}")
            return {'dpo_loss': 0.0}
        
        # Sample batch of preference pairs
        batch_indices = np.random.choice(len(self.preference_pairs), batch_size, replace=False)
        batch_pairs = [self.preference_pairs[i] for i in batch_indices]
        
        # Prepare batch data
        preferred_states = []
        preferred_actions = []
        less_preferred_states = []
        less_preferred_actions = []
        
        for preferred_traj, less_preferred_traj in batch_pairs:
            # Sample random timesteps from each trajectory
            if len(preferred_traj) > 0:
                idx = np.random.randint(len(preferred_traj))
                state, action = preferred_traj[idx]
                preferred_states.append(state)
                preferred_actions.append(action)
            
            if len(less_preferred_traj) > 0:
                idx = np.random.randint(len(less_preferred_traj))
                state, action = less_preferred_traj[idx]
                less_preferred_states.append(state)
                less_preferred_actions.append(action)
        
        if len(preferred_states) == 0 or len(less_preferred_states) == 0:
            logger.warning("No valid preference data in batch")
            return {'dpo_loss': 0.0}
        
        # Convert to tensors
        preferred_states = torch.FloatTensor(np.array(preferred_states)).to(self.device)
        preferred_actions = torch.FloatTensor(np.array(preferred_actions)).to(self.device)
        less_preferred_states = torch.FloatTensor(np.array(less_preferred_states)).to(self.device)
        less_preferred_actions = torch.FloatTensor(np.array(less_preferred_actions)).to(self.device)
        
        # Compute loss
        loss = self.compute_preference_loss(
            preferred_states, preferred_actions,
            less_preferred_states, less_preferred_actions
        )
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'dpo_loss': loss.item()}
    
    def update_reference_policy(self):
        """Update reference policy to current policy (optional, for curriculum learning)."""
        self.reference_policy.load_state_dict(self.policy.state_dict())
        logger.info("Reference policy updated to current policy")
    
    def save_model(self, path: str):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'reference_policy_state_dict': self.reference_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'beta': self.beta,
        }, path)
    
    def load_model(self, path: str):
        """Load model."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.reference_policy.load_state_dict(checkpoint['reference_policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.beta = checkpoint.get('beta', self.beta)

# Example usage functions
def create_dpo_from_demonstrations(demonstrations: List[List[Tuple[np.ndarray, np.ndarray]]],
                                 constraint_functions: List[callable],
                                 constraint_limits: List[float],
                                 state_dim: int,
                                 action_dim: int) -> DPOLearner:
    """
    Create and initialize DPO learner from demonstrations and constraints.
    
    Args:
        demonstrations: List of trajectories
        constraint_functions: List of constraint functions
        constraint_limits: List of constraint limits
        state_dim: State dimension
        action_dim: Action dimension
        
    Returns:
        Initialized DPO learner with preference data
    """
    dpo_learner = DPOLearner(state_dim, action_dim)
    
    # Create preference data from demonstrations
    preference_pairs = dpo_learner.create_constraint_preference_data(
        demonstrations, constraint_functions, constraint_limits
    )
    
    # Add preference pairs to learner
    for preferred_traj, less_preferred_traj in preference_pairs:
        dpo_learner.add_preference_pair(preferred_traj, less_preferred_traj)
    
    logger.info(f"Created {len(preference_pairs)} preference pairs from {len(demonstrations)} demonstrations")
    
    return dpo_learner 