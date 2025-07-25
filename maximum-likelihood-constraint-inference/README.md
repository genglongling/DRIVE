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
