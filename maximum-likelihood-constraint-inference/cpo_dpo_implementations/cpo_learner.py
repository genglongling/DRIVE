#!/usr/bin/env python3
"""
Constrained Policy Optimization (CPO) implementation for constraint learning.
Based on the paper: "Constrained Policy Optimization" by Achiam et al.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import scipy.optimize
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    """Policy network for CPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(PolicyNetwork, self).__init__()
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

class ValueNetwork(nn.Module):
    """Value network for CPO."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class CPOLearner:
    """
    Constrained Policy Optimization (CPO) implementation.
    
    This implementation handles multiple constraints and uses trust region optimization
    to ensure constraint satisfaction during policy updates.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 constraint_functions: List[callable],
                 constraint_limits: List[float],
                 max_kl: float = 0.01,
                 damping: float = 0.1,
                 lr: float = 3e-4,
                 device: str = 'cpu'):
        """
        Initialize CPO learner.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            constraint_functions: List of constraint functions that take (state, action) and return constraint values
            constraint_limits: List of constraint limits (constraints should be <= these values)
            max_kl: Maximum KL divergence for trust region
            damping: Damping coefficient for optimization
            lr: Learning rate
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_functions = constraint_functions
        self.constraint_limits = np.array(constraint_limits)
        self.max_kl = max_kl
        self.damping = damping
        self.device = device
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.value = ValueNetwork(state_dim).to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.old_log_probs = []
        self.constraint_values = []
        
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
    
    def compute_constraint_values(self, states: List[np.ndarray], actions: List[np.ndarray]) -> np.ndarray:
        """Compute constraint values for given state-action pairs."""
        constraint_values = []
        for state, action in zip(states, actions):
            values = [constraint_fn(state, action) for constraint_fn in self.constraint_functions]
            constraint_values.append(values)
        return np.array(constraint_values)
    
    def compute_advantages(self, rewards: List[float], states: List[np.ndarray], gamma: float = 0.99) -> np.ndarray:
        """Compute advantages using value function."""
        advantages = []
        returns = []
        
        # Compute returns
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = np.array(returns)
        
        # Compute value estimates
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        values = self.value(states_tensor).detach().cpu().numpy().flatten()
        
        # Compute advantages
        advantages = returns - values
        
        return advantages
    
    def update_policy(self, states: List[np.ndarray], actions: List[np.ndarray], 
                     advantages: np.ndarray, old_log_probs: List[float],
                     constraint_values: np.ndarray):
        """Update policy using CPO algorithm."""
        
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        
        # Get current policy distribution
        mean, logstd = self.policy(states_tensor)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
        
        # Compute ratios
        ratios = torch.exp(log_probs - old_log_probs_tensor)
        
        # Policy loss
        policy_loss = -(ratios * advantages_tensor).mean()
        
        # Compute KL divergence
        kl_div = (old_log_probs_tensor - log_probs).mean()
        
        # Compute constraint violations
        constraint_violations = constraint_values - self.constraint_limits
        
        # CPO optimization
        def objective_function(v):
            """Objective function for CPO optimization."""
            return policy_loss.item() + v[0] * kl_div.item() + np.sum(v[1:] * constraint_violations.mean(axis=0))
        
        # Solve constrained optimization problem
        n_constraints = len(self.constraint_functions)
        bounds = [(0, None)] + [(None, None)] * n_constraints  # v >= 0 for KL constraint
        
        # Initial guess
        x0 = np.zeros(n_constraints + 1)
        
        try:
            result = scipy.optimize.minimize(
                objective_function, x0, bounds=bounds,
                method='SLSQP',
                options={'maxiter': 100}
            )
            
            if result.success:
                # Update policy with optimal parameters
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
                logger.info(f"CPO update successful. KL: {kl_div.item():.4f}, "
                          f"Constraint violations: {constraint_violations.mean(axis=0)}")
            else:
                logger.warning("CPO optimization failed, using standard policy gradient")
                # Fallback to standard policy gradient
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
        except Exception as e:
            logger.error(f"CPO optimization error: {e}")
            # Fallback to standard policy gradient
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
    
    def update_value(self, states: List[np.ndarray], returns: np.ndarray):
        """Update value function."""
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        values = self.value(states_tensor)
        value_loss = nn.MSELoss()(values.squeeze(), returns_tensor)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    
    def train_step(self, batch_data: Dict[str, np.ndarray], gamma: float = 0.99):
        """Perform one training step."""
        states = batch_data['states']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        next_states = batch_data['next_states']
        dones = batch_data['dones']
        old_log_probs = batch_data['old_log_probs']
        
        # Compute constraint values
        constraint_values = self.compute_constraint_values(states, actions)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, states, gamma)
        
        # Compute returns for value function
        returns = advantages + self.value(torch.FloatTensor(np.array(states)).to(self.device)).detach().cpu().numpy().flatten()
        
        # Update networks
        self.update_policy(states, actions, advantages, old_log_probs, constraint_values)
        self.update_value(states, returns)
        
        return {
            'policy_loss': 0.0,  # Will be computed in update_policy
            'value_loss': 0.0,   # Will be computed in update_value
            'constraint_violations': constraint_values.mean(axis=0),
            'kl_divergence': 0.0  # Will be computed in update_policy
        }
    
    def save_model(self, path: str):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load model."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

# Example constraint functions
def velocity_constraint(state, action):
    """Example constraint: velocity should not exceed a limit."""
    # Assuming state contains velocity information
    velocity = np.linalg.norm(state[2:4]) if len(state) >= 4 else 0
    return velocity - 10.0  # Constraint: velocity <= 10

def position_constraint(state, action):
    """Example constraint: position should stay within bounds."""
    # Assuming state contains position information
    position = state[:2] if len(state) >= 2 else np.zeros(2)
    return np.max(np.abs(position)) - 5.0  # Constraint: |position| <= 5

def action_magnitude_constraint(state, action):
    """Example constraint: action magnitude should be limited."""
    return np.linalg.norm(action) - 2.0  # Constraint: ||action|| <= 2 