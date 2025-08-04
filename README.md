# DRIVE: Dynamic Rule Inference and Verified Evaluation for Constraint-Aware Autonomous Driving

<p align="center">
  ⬇️ <a href="https://github.com/genglongling/DRIVE">Github</a>  
  📃 <a href="https://arxiv.org/abs/2502.18836">Paper</a>  
  🌐 <a href="https://example.com/project">InD Dataset</a>
  🌐 <a href="https://example.com/project">HighD Dataset</a>
   🌐 <a href="https://example.com/project">RounD Dataset</a>
</p>

DRIVE is the first goal-based, mutli-objective, optimization framework for Constraint-Aware Autonomous Driving. DRIVE includes:
1) multi-objective supports: minimization of time, distance, effort, jerk, and lane change (in future).
2) constraint inference module:
   - using self-defined NN.
3) trajectory generation module:
   - using ICL + Convex Optimization (CO), Beam Search (BS), Markov Decision Process (MDP), Constrained Policy Optimization, Direct Preference Optimization (in future), PPO (in future).

--- DRIVE Authors

This repository is based on the [MATLAB implementation](https://drive.google.com/drive/folders/1h2J7o4w4J0_dpldTRpFu_jWQR8CkBbXw) provided by the authors of the following paper:
* [Scobee, Dexter RR, and S. Shankar Sastry. "Maximum Likelihood Constraint Inference for Inverse Reinforcement Learning." International Conference on Learning Representations. 2019.](https://openreview.net/forum?id=BJliakStvH)


Notes
=====
* OpenGL is required.
* For creating GIFs, install Imagemagick (`convert`) and `gifsicle`. Ensure that they can be accessed from command-line.
* `inD/000_background.png` has been created from `inD/00_background.png` using GIMP.
* `inD/00_*` are the zeroth track files from inD dataset (`data` folder in the inD dataset).
* `sim.py` contains code to read inD dataset as well as visualize through Pyglet.
* `main.py` contains code to do constraint inference on the inD dataset.
* `gridworld.py` contains code to do constraint inference on the Gridworld example from the MLCI paper.

Workflow
========
* To install pip requirements, run `pip3 install -r requirements.txt`
* To run Gridworld constraint inference example from the MLCI paper, run `python3 -B gridworld.py`
* To run the inD example,
    * Generate `pickles/trajectories.pickle` by running `python3 -B sim.py --generate_discrete_data`
    * Generate `pickles/constraints.pickle` and `pickles/new_demonstrations.pickle` by running `python3 -B main.py --do_constraint_inference`. To also generate policy plots, add `--policy_plot` flag as well.
    * Visualize the constraints and the dataset by running `python3 -B sim.py --visualize`.
    * (Optional) Produce GIFs by running `python3 -B sim.py --create_gifs`. This will create `frames.gif` and `policy.gif`.
    * Visualize the constraints and the generated demonstrations (from the final MDP) by running `python3 -B sim.py --visualize --show_new_demos`.
    * (Optional) Produce GIF by running `python3 -B sim.py --create_gifs --show_new_demos`. This will create `demos.gif`.
* To run the inD example in the multi-goal setting, run `sim.py` and `main.py` as in the previous step, but with `--multi_goal` flag as well.

Dataset Visualization
====================
* The repository includes the [drone-dataset-tools](https://github.com/drone-dataset-tools/drone-dataset-tools) for interactive visualization of the inD dataset.
* To use the interactive GUI visualization:
    * Navigate to the drone-dataset-tools directory: `cd drone-dataset-tools-master/src`
    * Run the fixed visualization script: `python3 run_track_visualization_fixed.py --dataset_dir /path/to/inD-dataset-v1.0/data --dataset ind --recording 00`
    * The GUI provides:
        - Interactive frame navigation with slider
        - Play/pause controls for animation
        - Vehicle visualization with bounding boxes and IDs
        - Background image display
        - Support for all inD dataset recordings (00-32)
* Alternative visualization options:
    - Use `--show_trajectory True` to display vehicle trajectories
    - Use `--annotate_track_id True` to show vehicle IDs
    - Use `--show_bounding_box True` to display vehicle bounding boxes
    - Use `--show_orientation True` to show vehicle orientation

Method Comparison
================

This repository implements and compares multiple constraint learning and optimization approaches for autonomous driving scenarios. Below is a detailed comparison of the different methods:

## Constraint Learning and Optimization Methods

| Method Category | Method Name | Learning Process | Constraint Handling | Optimization Type | Key Advantages | Key Limitations |
|----------------|-------------|------------------|-------------------|------------------|----------------|-----------------|
| **Traditional Planning** | Beam Search | No learning | Hard constraints | Search-based | Guaranteed optimality, interpretable | Computationally expensive, requires known constraints |
| **Constraint Learning** | MDP-ICL | Single-stage learning | Learned constraints | Inverse learning | Learns unknown constraints from data | Requires demonstration data, assumes MDP structure |
| **Two-Phase Optimization (ours)** | ICL + Convex | Two-stage: Learn → Optimize | Learned constraints | Convex optimization | Learns complex constraints, data-driven | Two separate optimizations, high computational cost for first optimization (training) + low computational cost for second optimization (deployment)|
| **Constrained RL** | CPO | Single-stage integrated | Soft pre-defined constraints | Trust region optimization | Efficient, handles multiple constraints | Requires known constraint functions |
| **Preference Learning** | DPO | Single-stage preference-based | Implicit through preferences | Preference optimization | Sample-efficient, no explicit rewards | Requires preference data, less interpretable |

## Detailed Method Characteristics

### Traditional Planning: Beam Search
- **Process**: Exhaustive search through action space
- **Constraint Source**: Pre-defined hard constraints
- **Optimization**: Branch-and-bound search
- **Use Case**: When constraints are known and optimality is required
- **Computational Complexity**: Exponential in search depth

### Constraint Learning: MDP-ICL
- **Process**: Inverse learning from demonstrations
- **Constraint Source**: Learned from trajectory data
- **Optimization**: Maximum likelihood estimation
- **Use Case**: Learning unknown constraints from expert demonstrations
- **Computational Complexity**: Linear in demonstration size

### Two-Phase Optimization: ICL + Convex (ours)
- **Process**: 
  1. **Phase 1**: Learn constraints using ICL
  2. **Phase 2**: Optimize policy with learned constraints
- **Constraint Source**: Learned from demonstrations
- **Optimization**: Convex optimization with hard constraints
- **Use Case**: Complex constraint discovery with guaranteed satisfaction
- **Computational Complexity**: Two-stage optimization

### Constrained RL: CPO
- **Process**: Integrated policy and constraint learning
- **Constraint Source**: Pre-defined constraint functions
- **Optimization**: Trust region with soft constraint penalties
- **Use Case**: Real-time policy learning with known constraints
- **Computational Complexity**: Single-stage optimization

### Preference Learning: DPO
- **Process**: Learning from preference pairs
- **Constraint Source**: Implicit through preference data
- **Optimization**: Direct preference optimization
- **Use Case**: Learning from human preferences without explicit rewards
- **Computational Complexity**: Linear in preference pairs

## Performance Comparison Metrics

| Metric | Beam Search | MDP-ICL | ICL + Convex (ours)| CPO | DPO |
|--------|-------------|---------|--------------|-----|-----|
| **Constraint Violation Rate** | Very Low | Low-Medium | Low | Medium | Medium-High |
| **Computational Efficiency** | Low | High | Medium | High | High |
| **Sample Efficiency** | N/A | Medium | Medium | High | Very High |
| **Interpretability** | Very High | High | High | Medium | Low |
| **Scalability** | Low | Medium | Medium | High | High |
| **Real-time Capability** | No | No | No | Yes | Yes |
| **Generalization** |  |  |  |  |  |

## Method Selection Guidelines

### Choose Beam Search when:
- Constraints are known and well-defined
- Optimality is required
- Computational resources are sufficient
- Interpretability is crucial

### Choose MDP-ICL when:
- Constraints are unknown but can be learned from demonstrations
- MDP structure is appropriate
- Demonstration data is available
- Single-stage learning is preferred

### Choose ICL + Convex (ours) when:
- Complex constraint relationships exist
- Hard constraint satisfaction is required
- Demonstration data is available
- Two-stage optimization is acceptable

### Choose CPO when:
- Constraint functions are known a priori
- Real-time policy updates are needed
- Multiple constraint types need to be handled
- Soft constraint handling is acceptable

### Choose DPO when:
- Preference data is available
- No explicit reward function exists
- Sample efficiency is crucial
- Human-in-the-loop learning is desired

## Experimental Setup

The repository includes comprehensive experimental comparison framework in `cpo_dpo_implementations/`:

```bash
# Run experimental comparison
cd cpo_dpo_implementations
python experimental_comparison.py

# This will compare all methods on:
# - Constraint violation rates
# - Reward performance
# - Training time
# - Convergence speed
# - Sample efficiency
```

## 🔍 Constraint Inference Framework

This repository includes a comprehensive **Constraint Inference** framework that implements constraint learning and filtering for autonomous driving scenarios.

### 🎯 Constraint Inference Methods

| Component | Purpose | Key Features | File |
|-----------|---------|--------------|------|
| **Constraint Model Utils** | Neural network for constraint prediction | Transition prediction with collision avoidance | `constraint_model_utils.py` |
| **Constraint Filtered Training** | RL training with constraint filtering | Only learn from constraint-satisfying trajectories | `constraint_filtered_training.py` |

### 🚀 Usage

#### Constraint Model Utilities
```bash
# Load and use constraint model
python -c "
from constraint_model_utils import load_model, get_nearby_vehicles, create_collision_avoidance_features
model, metadata = load_model('./model_checkpoint/model_00_tracks_f357680b.pth')
print('Constraint model loaded successfully')
"
```

#### Constraint Filtered Training
```bash
# Run constraint-filtered RL training
python constraint_filtered_training.py
```

### 📊 Key Features

#### 1. Constraint Model Architecture
- **TransitionPredictionNN**: Neural network for predicting transition likelihood
- **Input Dimension**: 23 features (14 base + 9 collision avoidance)
- **Architecture**: 128 → 64 → 32 → 1 with batch normalization and dropout
- **Activation**: ReLU with Xavier weight initialization

#### 2. Collision Avoidance Features
- **Nearby Vehicle Detection**: Identifies vehicles within proximity
- **Distance Calculation**: Computes distances to nearby vehicles
- **Relative Velocity**: Calculates relative motion between vehicles
- **Time to Collision**: Estimates collision risk using TTC
- **Feature Vector**: 9-dimensional collision avoidance features

#### 3. Constraint Filtered Training
- **Objective 1**: Train RL agent only from constraint-satisfying trajectories
- **Validation Ratio**: Minimum 70% valid experiences required for policy updates
- **Violation Tracking**: Monitors constraint violations during training
- **Episode Filtering**: Skips episodes with excessive violations (>60%)
- **Realistic Goals**: Extracts goals from actual trajectory endpoints

### 🔧 Implementation Details

#### Constraint Model Utils (`constraint_model_utils.py`)

##### Model Loading
```python
def load_model(filepath):
    """Load trained constraint model with metadata"""
    model_data = torch.load(filepath, map_location='cpu')
    model = TransitionPredictionNN(input_dim=model_data['model_config']['input_dim'])
    model.load_state_dict(model_data['model_state_dict'])
    return model, model_data
```

##### Nearby Vehicle Detection
```python
def get_nearby_vehicles(df, ego_track_id, frame, max_neighbors=5):
    """Get nearby vehicles for collision avoidance features"""
    frame_data = df[df['frame'] == frame]
    other_vehicles = frame_data[frame_data['trackId'] != ego_track_id]
    
    # Calculate distances and sort by proximity
    distances = []
    for _, vehicle in other_vehicles.iterrows():
        distance = np.linalg.norm([vehicle['xCenter'], vehicle['yCenter']] - ego_pos)
        distances.append((distance, vehicle))
    
    return sorted(distances, key=lambda x: x[0])[:max_neighbors]
```

##### Collision Avoidance Features
```python
def create_collision_avoidance_features(current_state, next_state, other_vehicles, current_frame, next_frame):
    """Create 9-dimensional collision avoidance feature vector"""
    collision_features = []
    
    for vehicle in other_vehicles:
        # Calculate distance features
        current_distance = np.linalg.norm(ego_current_pos - vehicle_current_pos)
        next_distance = np.linalg.norm(ego_next_pos - vehicle_next_pos)
        
        # Calculate relative velocity and TTC
        relative_vel = ego_next_vel - vehicle_current_vel
        relative_speed = np.linalg.norm(relative_vel)
        ttc = next_distance / relative_speed if relative_speed > 0.1 else 100.0
        
        collision_features.extend([current_distance, next_distance, relative_speed, ttc, ...])
    
    return collision_features[:9]  # Return exactly 9 features
```

#### Constraint Filtered Training (`constraint_filtered_training.py`)

##### Training Process
```python
def create_constraint_filtered_training_data(csv_file, constraint_model_path=None, num_episodes=1000, min_valid_ratio=0.7):
    """Train RL agent with constraint filtering"""
    
    # Initialize RL agent with constraint model
    rl_agent = ConstraintGuidedRL(
        state_dim=23,
        action_dim=4,
        constraint_model_path=constraint_model_path
    )
    
    # Training with constraint filtering
    for episode in range(num_episodes):
        # Generate episode
        episode_reward, constraint_violations = rl_agent.train_episode(
            initial_state, df, max_steps=50, goal_pos=episode_goal
        )
        
        # Only use constraint-satisfying episodes for learning
        if constraint_violations <= max_steps * 0.6:  # Allow up to 60% violations
            valid_episodes += 1
            rl_agent.training_stats['episode_rewards'].append(episode_reward)
            print(f"Episode {episode + 1}: ✅ Valid")
        else:
            print(f"Episode {episode + 1}: ❌ Skipped due to violations")
```

##### Evaluation Process
```python
def evaluate_constraint_filtered_agent(rl_agent, df, num_episodes=50, max_steps=100):
    """Evaluate constraint-filtered RL agent"""
    
    for episode in range(num_episodes):
        # Generate trajectory
        trajectory, actions, rewards = generate_rl_trajectories(
            rl_agent, initial_state, df, max_steps=max_steps
        )
        
        # Count constraint violations
        constraint_violations = sum(1 for r in rewards if r <= -1000)
        
        # Only include episodes with no constraint violations
        if constraint_violations == 0:
            valid_evaluations += 1
            episode_rewards.append(total_reward)
            episode_trajectories.append(trajectory_data)
```

### 📈 Performance Metrics

#### Training Metrics
- **Valid Episodes**: Number of constraint-satisfying episodes
- **Violation Rate**: Percentage of episodes with constraint violations
- **Average Reward**: Mean reward across valid episodes
- **Training Progress**: Episode-by-episode learning progress

#### Evaluation Metrics
- **Constraint Satisfaction**: Percentage of evaluation episodes without violations
- **Travel Time**: Time to reach goal for valid trajectories
- **Final Distance**: Distance to goal at episode end
- **Trajectory Quality**: Smoothness and efficiency of generated paths

### 🎨 Output Files

#### Constraint Filtered Training
- **Results JSON**: `constraint_filtered_results/constraint_filtered_results.json`
- **Trajectory CSV**: `constraint_filtered_results/constraint_filtered_trajectories.csv`
- **Training Statistics**: Episode rewards, violations, and performance metrics

#### Model Utilities
- **Model Loading**: Support for loading pre-trained constraint models
- **Feature Extraction**: Collision avoidance feature generation
- **Vehicle Detection**: Nearby vehicle identification and tracking

### 🔬 Research Applications

This framework enables:
- **Constraint Learning**: Study of constraint inference from demonstrations
- **Safe RL Training**: Training RL agents with constraint satisfaction guarantees
- **Collision Avoidance**: Analysis of collision avoidance strategies
- **Performance Benchmarking**: Evaluation of constraint satisfaction methods
- **Real-world Validation**: Testing on actual driving data with safety constraints

## 🤖 RL with Constraints Framework

This repository includes a comprehensive **Reinforcement Learning with Constraints** framework that implements five different optimization approaches for autonomous driving scenarios.

### 🎯 RL with Constraints Methods

| Method | Optimization Approach | Constraint Handling | Key Features | Model File | Test File |
|--------|---------------------|-------------------|--------------|------------|-----------|
| **Beam Search** | Search-based planning | Hard constraints | Optimal path planning | `rl_with_constraints_beamsearch.py` | `test_rl_with_constraints_beamsearch.py` |
| **Convex** | Convex optimization | Soft constraints | Smooth trajectory optimization | `rl_with_constraints_convex.py` | `test_rl_with_constraints_convex.py` |
| **MDP** | Markov Decision Process | Learned constraints | State transition modeling | `rl_with_constraints_mdp.py` | `test_rl_with_constraints_mdp.py` |
| **CPO** | Constrained Policy Optimization | Constraint penalties | Trust region optimization | `rl_with_constraints_cpo.py` | `test_rl_with_constraints_cpo.py` |
| **DPO** | Direct Preference Optimization | Preference learning | Human preference alignment | `rl_with_constraints_dpo.py` | `test_rl_with_constraints_dpo.py` |

### 🚀 Usage

#### Training and Testing
```bash
# Train and test Beam Search approach
python test_rl_with_constraints_beamsearch.py

# Train and test Convex optimization approach
python test_rl_with_constraints_convex.py

# Train and test MDP approach
python test_rl_with_constraints_mdp.py

# Train and test CPO approach
python test_rl_with_constraints_cpo.py

# Train and test DPO approach
python test_rl_with_constraints_dpo.py
```

#### Model Files
Each method has its own implementation file:
- **`rl_with_constraints_beamsearch.py`**: Beam search with collision avoidance
- **`rl_with_constraints_convex.py`**: Convex optimization for constraint satisfaction
- **`rl_with_constraints_mdp.py`**: MDP-based planning with transition models
- **`rl_with_constraints_cpo.py`**: Constrained policy optimization
- **`rl_with_constraints_dpo.py`**: Direct preference optimization

#### Test Files
Each method has a corresponding test file that:
- Splits dataset into training (80%) and generation (20%) sets
- Trains the RL agent with constraint model integration
- Generates trajectories for specific test tracks (233, 235, 248)
- Saves real trajectory data for comparison
- Creates comprehensive visualizations

### 📊 Output Files

Each method generates:
- **Trained Model**: `rl_agent_trained_[method].pth`
- **Training History**: `[method]_rl_training_history.csv`
- **Episode Summary**: `[method]_rl_episode_summary.csv`
- **Generation Results**: `generation_results_[method].pkl`
- **Visualization**: Individual trajectory plots and comprehensive analysis

### 🎨 Visualization Features

Each method provides:
- **Trajectory Comparison**: Planned vs real ego car trajectories
- **Front Car Context**: Real front car trajectory visualization
- **Performance Metrics**: Position, velocity, and acceleration over time
- **Individual Plots**: Separate detailed plots for each trajectory
- **Statistical Analysis**: Comprehensive performance summary

### 🔧 Key Features

#### 1. Constraint Integration
- **Pre-trained Constraint Model**: Uses `model_checkpoint/model_00_tracks_f357680b.pth`
- **Constraint Satisfaction**: Each method handles constraints differently
- **Violation Tracking**: Monitors constraint violations during training

#### 2. Dataset Management
- **Training Set**: 80% of tracks for agent training
- **Generation Set**: 20% of tracks for trajectory generation
- **Specific Test Tracks**: Tracks 233, 235, 248 for detailed analysis
- **Real Trajectory Extraction**: Captures actual vehicle trajectories for comparison

#### 3. Performance Tracking
- **Training Rewards**: Episode-by-episode reward tracking
- **Constraint Violations**: Violation rate monitoring
- **Memory Usage**: Peak memory consumption tracking
- **Training Time**: Wall-clock training time measurement

#### 4. Method-Specific Optimizations

##### Beam Search
- **Search-based planning** with collision avoidance
- **Optimal path finding** with velocity constraints
- **Hard constraint satisfaction** through search pruning

##### Convex Optimization
- **Smooth trajectory optimization** using convex programming
- **Soft constraint handling** with regularization
- **Efficient optimization** with guaranteed convergence

##### MDP Planning
- **State transition modeling** for trajectory prediction
- **Learned constraint integration** in transition functions
- **Markov property exploitation** for efficient planning

##### CPO (Constrained Policy Optimization)
- **Trust region optimization** with constraint penalties
- **Soft constraint handling** through penalty functions
- **Real-time policy updates** with constraint awareness

##### DPO (Direct Preference Optimization)
- **Preference-based learning** without explicit rewards
- **Human preference alignment** through preference pairs
- **Sample-efficient learning** with preference data

### 📈 Performance Comparison

| Method | Constraint Handling | Training Speed | Memory Usage | Real-time Capability | Interpretability |
|--------|-------------------|----------------|--------------|---------------------|------------------|
| **Beam Search** | Hard constraints | Slow | High | No | Very High |
| **Convex** | Soft constraints | Medium | Medium | No | High |
| **MDP** | Learned constraints | Medium | Medium | No | High |
| **CPO** | Constraint penalties | Fast | Low | Yes | Medium |
| **DPO** | Preference learning | Fast | Low | Yes | Low |

### 🎯 Research Applications

This framework enables:
- **Method Comparison**: Direct comparison of different RL approaches
- **Constraint Learning**: Study of constraint inference methods
- **Trajectory Planning**: Analysis of planning algorithms
- **Performance Benchmarking**: Comprehensive evaluation metrics
- **Real-world Validation**: Testing on actual driving data

## **📜 Repository Structure**

```
EE364B/
├── README.md                                    # Main project documentation
├── basic.py                                     # Basic utility functions
└── maximum-likelihood-constraint-inference/     # Main project directory
    ├── README.md                               # Project-specific documentation
    ├── basic.py                                # Basic utility functions
    
    # Core RL Implementation Files
    ├── rl_with_constraints_beamsearch.py       # Beam search RL implementation
    ├── rl_with_constraints_convex.py           # Convex optimization RL
    ├── rl_with_constraints_mdp.py              # MDP-based RL
    ├── rl_with_constraints_cpo.py              # CPO RL implementation
    ├── rl_with_constraints_dpo.py              # DPO RL implementation
    
    # Test and Evaluation Files
    ├── test_rl_with_constraints_beamsearch.py  # Beam search testing
    ├── test_rl_with_constraints_convex.py      # Convex optimization testing
    ├── test_rl_with_constraints_mdp.py         # MDP testing
    ├── test_rl_with_constraints_cpo.py         # CPO testing
    ├── test_rl_with_constraints_dpo.py         # DPO testing
    
    # Generalization Test Files
    ├── test_rl_with_constraints_convex_generalization_round.py  # rounD dataset
    ├── test_rl_with_constraints_convex_generalization_highd.py # highD dataset
    ├── test_rl_with_constraints_beamsearch_generalization_ind.py # inD beamsearch
    ├── test_rl_with_constraints_mdp_generalization_ind.py       # inD MDP
    └── test_rl_with_constraints_cpo_generalization_ind.py       # inD CPO
    
    # Constraint Inference Framework
    ├── constraint_model_utils.py               # Constraint model utilities
    ├── constraint_filtered_training.py         # Constraint-filtered RL training
    
    # Analysis and Visualization
    ├── trajectory_quality_analysis.py          # Trajectory quality metrics
    ├── constraint_quality_analysis.py          # Constraint violation analysis
    ├── comprehensive_analysis_report.py        # Comprehensive analysis
    ├── visualize_rl_with_constraints.py       # Main visualization script
    └── comprehensive_method_analysis_generalization.py # Generalization analysis
    
    # Documentation and Tables
    ├── hyperparameter_tables.tex              # Hyperparameter documentation
    ├── constraint_violations_convex_round_highd.tex # Generalization results
    └── scalability_complexity_analysis.md     # Complexity analysis
    
    # Dataset Directories
    ├── dataset/
    │   ├── inD/                               # inD dataset files
    │   ├── rounD/                             # rounD dataset files
    │   └── highD/                             # highD dataset files
    
    # Output Directories
    ├── trajectory_data/                        # Generated trajectory data
    ├── metrics/                                # Performance metrics
    ├── visualization/                          # Generated visualizations
    ├── visualization_output/                   # Analysis outputs
    ├── integrated_results/                     # Integrated analysis results
    ├── model_checkpoint/                       # Trained model checkpoints
    ├── pickles/                               # Pickled data files
    └── figures/                               # Generated figures
    
    # Environment and Dependencies
    ├── rl_convex_env/                         # Python virtual environment
    └── requirements.txt                       # Python dependencies
```

### Key Directories Explained

#### **Core Implementation (`/`)**
- **RL Method Files**: Individual implementations for each RL approach
- **Test Files**: Comprehensive testing and evaluation scripts
- **Constraint Framework**: Constraint inference and filtering utilities

#### **Dataset Management (`/dataset/`)**
- **inD/**: Intersection dataset with traffic scenarios
- **rounD/**: Roundabout dataset for generalization testing
- **highD/**: Highway dataset for generalization testing

#### **Output Management**
- **trajectory_data/**: Generated trajectories and training history
- **visualization/**: Plots and analysis figures
- **metrics/**: Performance metrics and statistics
- **model_checkpoint/**: Trained model files

#### **Analysis and Documentation**
- **Analysis Scripts**: Quality metrics, constraint analysis, comprehensive reports
- **Documentation**: LaTeX tables, complexity analysis, hyperparameter documentation
- **Visualization**: Interactive plots and trajectory comparisons

## **📜 Citation**  

If you find this repository helpful, please cite the following paper:  

```
DRIVE: Dynamic Rule Inference and Verified Evaluation for Constraint-Aware Autonomous Driving
Anonymous Author(s)  
```

---
