# CPO and DPO Implementations for Constraint Learning

This directory contains implementations of **Constrained Policy Optimization (CPO)** and **Direct Preference Optimization (DPO)** for constraint learning, designed to extend your experimental comparison beyond the existing methods.

## Overview

### CPO (Constrained Policy Optimization)
- **Purpose**: Learn policies that satisfy constraints during training
- **Key Feature**: Handles multiple constraints with trust region optimization
- **Advantage**: No need for separate constraint optimization step
- **Use Case**: Replace your "RL + Convex Optimization" approach

### DPO (Direct Preference Optimization)
- **Purpose**: Learn from preference data without explicit reward functions
- **Key Feature**: Creates preference pairs from constraint satisfaction
- **Advantage**: More sample-efficient than RLHF
- **Use Case**: Alternative to reward learning + policy optimization

## Files

- `cpo_learner.py` - CPO implementation with constraint handling
- `dpo_learner.py` - DPO implementation with preference learning
- `experimental_comparison.py` - Comprehensive comparison framework
- `requirements.txt` - Dependencies for the implementations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic CPO Usage

```python
from cpo_learner import CPOLearner, velocity_constraint, position_constraint

# Define constraints
constraint_functions = [velocity_constraint, position_constraint]
constraint_limits = [10.0, 5.0]  # velocity <= 10, position <= 5

# Initialize CPO learner
cpo_learner = CPOLearner(
    state_dim=8,
    action_dim=2,
    constraint_functions=constraint_functions,
    constraint_limits=constraint_limits
)

# Get action from policy
state = np.random.randn(8)
action, log_prob = cpo_learner.get_action(state)

# Train with batch data
batch_data = {
    'states': [...],
    'actions': [...],
    'rewards': [...],
    'next_states': [...],
    'dones': [...],
    'old_log_probs': [...]
}
metrics = cpo_learner.train_step(batch_data)
```

### Basic DPO Usage

```python
from dpo_learner import DPOLearner, create_dpo_from_demonstrations

# Create demonstrations (list of trajectories)
demonstrations = [
    [(state1, action1), (state2, action2), ...],  # Trajectory 1
    [(state1, action1), (state2, action2), ...],  # Trajectory 2
    ...
]

# Initialize DPO learner from demonstrations
dpo_learner = create_dpo_from_demonstrations(
    demonstrations, constraint_functions, constraint_limits,
    state_dim=8, action_dim=2
)

# Train DPO
for episode in range(1000):
    metrics = dpo_learner.train_step(batch_size=32)
    if episode % 100 == 0:
        print(f"Episode {episode}: Loss = {metrics['dpo_loss']:.4f}")
```

### Full Experimental Comparison

```python
from experimental_comparison import ExperimentalComparison

# Initialize comparison
comparison = ExperimentalComparison(state_dim=8, action_dim=2)

# Run all experiments
results = comparison.run_all_experiments(n_episodes=1000)

# Plot and save results
comparison.plot_results("comparison_results.png")
comparison.save_results("comparison_results.csv")
```

## Integration with Existing Methods

The experimental comparison framework is designed to work alongside your existing methods:

1. **Beam Search** - Traditional search-based approach
2. **MDP-ICL** - Inverse Constraint Learning with MDPs  
3. **RL + Convex Optimization** - Your current approach
4. **CPO** - Constrained policy optimization (NEW)
5. **DPO** - Direct preference optimization (NEW)

## Key Features

### CPO Features
- ✅ Multiple constraint handling
- ✅ Trust region optimization
- ✅ Fallback to standard policy gradient
- ✅ Constraint violation monitoring
- ✅ KL divergence control

### DPO Features
- ✅ Preference-based learning
- ✅ Automatic preference pair generation from constraints
- ✅ Reference policy management
- ✅ Temperature parameter control
- ✅ Batch training support

## Metrics Tracked

Both implementations track:
- **Constraint Violations**: Rate of constraint violations
- **Rewards**: Performance rewards
- **Training Time**: Computational cost
- **Convergence Episodes**: Learning efficiency

## Example Constraint Functions

The implementations include example constraint functions:

```python
def velocity_constraint(state, action):
    """Velocity should not exceed limit."""
    velocity = np.linalg.norm(state[2:4]) if len(state) >= 4 else 0
    return velocity - 10.0  # Constraint: velocity <= 10

def position_constraint(state, action):
    """Position should stay within bounds."""
    position = state[:2] if len(state) >= 2 else np.zeros(2)
    return np.max(np.abs(position)) - 5.0  # Constraint: |position| <= 5

def action_magnitude_constraint(state, action):
    """Action magnitude should be limited."""
    return np.linalg.norm(action) - 2.0  # Constraint: ||action|| <= 2
```

## Research Contributions

These implementations enable you to:

1. **Extend your experimental comparison** with modern RL methods
2. **Compare constraint handling approaches** (separate vs. integrated)
3. **Evaluate preference-based learning** vs. reward-based learning
4. **Provide comprehensive analysis** of constraint learning methods

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run experimental comparison**: `python experimental_comparison.py`
3. **Integrate with your existing codebase** by importing the learners
4. **Customize constraint functions** for your specific domain
5. **Extend the comparison** with additional metrics or methods

## References

- **CPO**: Achiam, J., et al. "Constrained Policy Optimization." ICML 2017.
- **DPO**: Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023. 