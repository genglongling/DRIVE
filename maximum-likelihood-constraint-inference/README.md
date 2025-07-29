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

Integrated Methods Comparison with DVnetworks
============================================

This repository provides a comprehensive integration framework that combines **DVnetworks (Directional Velocity Networks)** with all baseline methods for constraint learning and optimization in autonomous driving.

## 🎯 Integration Overview

The integration framework combines:
- **DVnetworks**: Specialized neural networks for trajectory prediction
- **Baseline Methods**: All existing constraint learning and optimization approaches
- **Performance Tracking**: Comprehensive metrics for comparison
- **Visualization**: Rich plotting and analysis tools

## 🏗️ Integration Architecture

### Integrated Methods

| Method | Integration Approach | Key Features | Distance Usage | Velocity Usage | Acceleration Usage | Implementation Source |
|--------|-------------------|--------------|----------------|----------------|-------------------|-------------------|
| **Beam Search** | Use DVnetworks predictions to guide search | Optimal path planning with learned velocity patterns | Position + Closest car | Current + Predicted | Calculated | `main_beamsearch.py` |
| **MDP-ICL** | Learn constraints from DVnetworks-predicted trajectories | Enhanced constraint inference with velocity predictions | Position + Closest car | Current + Predicted | Calculated | `main_ICL.py` + `mdp.py` |
| **ICL + Convex** | Use DVnetworks in constraint inference phase | Our method with improved trajectory understanding | Position + Closest car | Current + Predicted | Calculated | `main_EFLCE.py` |
| **CPO** | Use DVnetworks as part of state representation | Constrained policy optimization with velocity awareness | Position + Closest car | Current + Predicted + Policy | Calculated | `cpo_dpo_implementations/` |
| **DPO** | Use DVnetworks for preference-based learning | Direct preference optimization with trajectory context | Position + Closest car | Current + Predicted + Policy | Calculated | `cpo_dpo_implementations/` |

### DVnetworks Integration

#### Direction Network
- **Input**: `[x, y, velocity_angle]` (current position and velocity direction)
- **Output**: `velocity_angle` at t+40 (predicted future direction)
- **Integration**: Guides trajectory planning in all methods

#### Amplitude Network  
- **Input**: `[x, y, velocity_magnitude, closest_car_distance, moving_towards_ego]`
- **Output**: `velocity_magnitude` at t+40 (predicted future speed)
- **Integration**: Provides velocity constraints and collision avoidance

## 📊 Tracked Metrics

### 1. Trajectory Data Storage
Each method stores comprehensive trajectory information:

```python
# Planned ego car trajectory
planned_ego_track_id: int
planned_ego_x, planned_ego_y: float  # Position
planned_ego_vx, planned_ego_vy: float  # Velocity
planned_ego_ax, planned_ego_ay: float  # Acceleration
planned_ego_frame: int  # Time step

# Real ego car trajectory (for comparison)
real_ego_track_id: int
real_ego_x, real_ego_y: float
real_ego_vx, real_ego_vy: float
real_ego_ax, real_ego_ay: float
real_ego_frame: int

# Real front car trajectory (for context)
real_front_track_id: int
real_front_x, real_front_y: float
real_front_vx, real_front_vy: float
real_front_ax, real_front_ay: float
real_front_frame: int

# Method identification
method_name: str  # 'beam_search', 'mdp_icl', 'icl_convex', 'cpo', 'dpo'
```

### 2. Performance Metrics
Comprehensive performance tracking for each method:

- **Training Time**: Wall-clock time for method training
- **Inference Time**: Time to generate trajectories
- **Peak Memory Usage**: Maximum RAM consumption (MB)
- **CPU Time**: Total CPU processing time
- **Memory Efficiency**: Performance per memory unit

## 🚀 Integration Usage

### 1. Installation

```bash
# Install integration dependencies
pip install -r integration_requirements.txt

# Ensure DVnetworks are trained
cd DVnetworks
python directional_velocity_networks.py
```

### 2. Available Versions

The framework provides two versions for different use cases:

#### Full Version (Recommended)
```bash
# Run comprehensive comparison with actual implementations
python integrated_methods_comparison.py

# Generate rich visualizations with plots
python visualize_integrated_results.py
```

**Features:**
- Imports actual implementations from existing files
- Uses real `beam_search()` function from `main_beamsearch.py`
- Uses `GridMDP` class from `main_ICL.py` and `mdp.py`
- Uses `TransitionPredictionNN` from `main_EFLCE.py`
- Uses `CPOLearner` and `DPOLearner` from `cpo_dpo_implementations/`
- Generates high-quality PNG plots and comprehensive analysis
- Requires: `numpy`, `pandas`, `matplotlib`, `seaborn`, `torch`, `psutil`

#### Simulated Version (Lightweight)
```bash
# Run lightweight comparison for testing
python integrated_methods_comparison_simulated.py

# Generate text-based summaries
python visualize_integrated_results_simulated.py
```

**Features:**
- Uses only basic Python libraries (no external dependencies)
- Simulates method behavior for testing and demonstration
- Generates text summaries and CSV files
- Fast execution with minimal memory usage
- Perfect for quick testing and environments with dependency issues

### 3. Run Integrated Comparison

```bash
# Run comprehensive comparison (Full Version)
python integrated_methods_comparison.py

# Or run lightweight version (Simulated)
python integrated_methods_comparison_simulated.py
```

This will:
- Load pre-trained DVnetworks (if available)
- Initialize all methods with actual implementations (Full) or simulations (Simulated)
- Run each method with DVnetworks integration
- Track all performance metrics
- Store trajectory data for visualization
- Generate comparison plots (Full) or text summaries (Simulated)

### 4. Visualize Results

```bash
# Create comprehensive visualizations (Full Version)
python visualize_integrated_results.py

# Or generate text summaries (Simulated Version)
python visualize_integrated_results_simulated.py
```

This generates:

**Full Version:**
- **Trajectory Comparison**: Visual comparison of all methods
- **Velocity Analysis**: Speed patterns across methods
- **Acceleration Analysis**: Acceleration patterns
- **Performance Metrics**: Training/inference time comparison
- **Method Summary**: Comprehensive comparison charts
- **Statistics Report**: Detailed numerical analysis

**Simulated Version:**
- **Text Summary**: Comprehensive text-based analysis
- **CSV Reports**: Trajectory and performance data in CSV format
- **Statistics Report**: Detailed numerical analysis in text format

## 📁 Integration Output Files

### Generated Data Files
- `integrated_trajectory_data.csv`: Complete trajectory data for all methods
- `integrated_trajectory_data.pkl`: Pickled trajectory data
- `integrated_results.csv`: Performance metrics summary
- `integrated_performance_comparison.png`: Performance comparison plots

### Visualization Output
- `visualization_output/trajectory_comparison.png`: Trajectory plots
- `visualization_output/velocity_comparison.png`: Velocity analysis
- `visualization_output/acceleration_comparison.png`: Acceleration analysis
- `visualization_output/performance_metrics.png`: Performance charts
- `visualization_output/method_comparison_summary.png`: Comprehensive summary
- `visualization_output/statistics_report.txt`: Detailed statistics

## 🔧 Integration Details

### Actual Implementation Integration

The Full Version integrates with actual implementations from existing files:

#### Beam Search Integration
```python
# Imports from main_beamsearch.py
from main_beamsearch import beam_search, TransitionPredictionNN

def run_beam_search_with_dvnetworks(self, initial_state, max_steps=100):
    # Use actual beam_search function
    a_values = np.array([[predicted_vx, predicted_vy]])
    next_state = beam_search(current_state, self.beam_search_model, a_values, delta_t=0.04, max_depth=10)
    
    # Store comprehensive trajectory data
    # Track performance metrics
```

#### MDP-ICL Integration
```python
# Imports from main_ICL.py and mdp.py
from main_ICL import GridMDP
from mdp import MDP, GridMDP as MDPGridMDP

def run_mdp_icl_with_dvnetworks(self, initial_state, max_steps=100):
    # Use actual GridMDP implementation
    if hasattr(self.methods['mdp_icl'], 'get_action'):
        next_state = self.methods['mdp_icl'].get_action(current_state)
    elif hasattr(self.methods['mdp_icl'], 'step'):
        next_state = self.methods['mdp_icl'].step(current_state)
```

#### ICL + Convex Integration
```python
# Imports from main_EFLCE.py
from main_EFLCE import TransitionPredictionNN as EFLCETransitionPredictionNN

def run_icl_convex_with_dvnetworks(self, initial_state, max_steps=100):
    # Use actual TransitionPredictionNN for convex optimization
    input_features = np.concatenate([current_state, closest_car_features])
    prediction = self.methods['icl_convex'](torch.tensor(input_features, dtype=torch.float32))
    next_state = self.convert_prediction_to_state(current_state, prediction)
```

#### CPO/DPO Integration
```python
# Imports from cpo_dpo_implementations/
from cpo_dpo_implementations.cpo_learner import CPOLearner
from cpo_dpo_implementations.dpo_learner import DPOLearner

def run_cpo_with_dvnetworks(self, initial_state, max_steps=100):
    # Use actual CPOLearner implementation
    self.methods['cpo'].train(demonstrations)
    next_state = self.methods['cpo'].get_action(current_state)
```

### Fallback Behavior

All methods include graceful fallback to simulated behavior if:
- Import errors occur (missing dependencies)
- Implementation errors occur (incompatible interfaces)
- Memory constraints are exceeded
- Performance issues arise

This ensures the framework remains functional even when actual implementations are unavailable.



## 📈 Integration Key Features

### 1. Comprehensive Data Tracking
- **Trajectory Storage**: Complete trajectory data for all methods
- **Performance Monitoring**: Real-time tracking of computational resources
- **Context Awareness**: Front car information for collision avoidance
- **Method Comparison**: Direct comparison across all approaches

### 2. DVnetworks Enhancement
- **Velocity Prediction**: Accurate velocity direction and magnitude prediction
- **Temporal Modeling**: 40-frame prediction horizon with smoothing
- **Context Integration**: Closest car distance and relative motion
- **Noise Reduction**: 40-frame smoothing for robust predictions

### 3. Performance Analysis
- **Memory Usage**: Peak memory consumption tracking
- **CPU Utilization**: Processing time analysis
- **Training Efficiency**: Method-specific training time comparison
- **Inference Speed**: Real-time capability assessment

### 4. Visualization Suite
- **Trajectory Plots**: Visual comparison of planned vs real trajectories
- **Performance Charts**: Training/inference time comparison
- **Velocity Analysis**: Speed pattern comparison
- **Acceleration Analysis**: Acceleration pattern comparison
- **Comprehensive Summary**: Radar charts and statistical analysis

## 🎯 Integration Research Contributions

### 1. Method Integration
- **First comprehensive integration** of DVnetworks with constraint learning methods
- **Enhanced trajectory prediction** for all baseline approaches
- **Improved constraint inference** with velocity awareness

### 2. Performance Benchmarking
- **Standardized comparison framework** for all methods
- **Comprehensive metrics tracking** for fair evaluation
- **Real-world performance analysis** with actual trajectory data

### 3. Visualization and Analysis
- **Rich visualization suite** for method comparison
- **Statistical analysis tools** for performance evaluation
- **Reproducible evaluation framework** for research community

## 🔮 Future Extensions

### 1. Additional Methods
- **PPO Integration**: Proximal Policy Optimization with DVnetworks
- **SAC Integration**: Soft Actor-Critic with velocity prediction
- **TD3 Integration**: Twin Delayed DDPG with trajectory awareness

### 2. Enhanced Metrics
- **Safety Metrics**: Collision avoidance performance
- **Efficiency Metrics**: Fuel consumption and energy efficiency
- **Comfort Metrics**: Passenger comfort and smoothness

### 3. Advanced Visualization
- **3D Trajectory Plots**: Three-dimensional trajectory visualization
- **Interactive Dashboards**: Real-time method comparison
- **Animation Support**: Animated trajectory comparison
