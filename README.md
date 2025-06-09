# Constraint Learning Convex Optimization for Autonomous Driving Vehicle (EE364B)

## Abstract
By utilizing Exponential Family Likelihood Estimation Constraint Learning (EFLCE), we aim to: (1) learn soft constraints with the varying types of constraints followed by different drivers, (2) perform state estimation to accurately assess the current driving context and (3) evaluate the performance of the learned agent across diverse driving scenarios.

## Introduction
As autonomous driving technology advances, learning from human driving behavior presents a valuable opportunity to improve the capabilities and safety of autonomous systems. While hard constraints (e.g., speed limits) can be explicitly programmed, soft constraints—like comfort, driving habits, or preferences—are more difficult to capture. Our research focuses on identifying and learning these nuanced constraints embedded in expert driving data using a neural network-based ICLR framework, integrating both hard and soft constraints for robust autonomous vehicle control.

## Related Work
Autonomous driving relies on machine learning and deep learning algorithms to navigate complex environments. Foundational approaches include Inverse Reinforcement Learning (IRL) and Inverse Constrained Reinforcement Learning (ICRL). Recent advancements incorporate Bayesian methods and neural networks to infer probabilistic distributions over constraints, improving adaptability and generalization.

## Dataset and Features
We use the HighD and inD datasets, which record vulnerable road users on German highways and intersections. The datasets provide detailed information about trajectories, positions, velocities, accelerations, and interactions. Preprocessing includes filtering, grouping, and aligning trajectories, and extracting kinematic and interaction features.

## Methodology
We predict the feasible score (reward, 0 to 1) for state transitions using a neural network, trained with a loss function that balances prediction accuracy and adherence to constraints. The EFLCE framework models the likelihood of state transitions under soft constraints, using maximum likelihood estimation. Reinforcement learning (Q-learning or Policy Gradient) is used to optimize trajectories, and convex optimization ensures efficient and feasible solutions.

### Key Equations
- Loss function: combines prediction error and constraint violation penalty
- EFLCE: models transition likelihood as a probabilistic distribution
- Reward: incentivizes safe and efficient trajectories
- Convex optimization: ensures constraints are satisfied during trajectory planning

## Experiments and Results
We evaluate our RL model using the inD dataset, focusing on:
- **Success Rate (Collision Avoidance):** Percentage of successful collision avoidance attempts
- **Average Reward:** Average reward per episode, encouraging smooth action and collision avoidance
- **Constraint Violation Rate:** Percentage of generated trajectories violating ICL constraints

| Metric                          | Beam Search + NN (Baseline) | MDP - ICL | EFLCE  |
|----------------------------------|-----------------------------|-----------|--------|
| Success Rate (Collision Avoid.)  | 98.25%                      | 98.12%    | 100%   |
| Average Reward                   | 0.5027                      | -0.4266   | 0.97   |
| Constraint Violation Rate        | 1.74%                       | 0.74%     | 0.21%  |

- **EFLCE outperforms** other methods in all metrics, achieving 100% collision avoidance, highest average reward, and lowest constraint violation rate.
- The system shows early convergence and high precision in constraint inference, with constraints aligning closely to actual obstacles after training.
- The model generalizes well to new scenarios, including different speeds and environments.

## Model Comparison
- **Baseline:** Three-layer NN with beam search for trajectory selection.
- **MDP:** Markov Decision Process for state transitions and constraint selection.
- **EFLCE:** Provides highest stability and robustness, with minimal variance and fast convergence.

## Factor Analysis
- **Sensitivity to Soft Constraints:** EFLCE captures stable velocity and acceleration constraints, outperforming beam search in stability.
- **Discrete to Continuous State:** EFLCE enables efficient state estimation without exhaustive search.
- **Applicability:** Achieves 0% collision rate on High-D dataset with front cars among 1000 test trajectories, showing strong generalization.

## Conclusion and Future Work
EFLCE demonstrates high accuracy, low constraint violation, and strong generalization for learning soft constraints in autonomous driving. The framework bridges the gap in current self-driving car problems by enabling safe, efficient, and human-like driving behavior. Future work will explore more complex scenarios, such as lane changing and multiple cars.

## References
- [1] Bachute and Subhade, Review of IRL algorithms
- [2] Abbeel and Ng, Apprenticeship Learning via IRL
- [3] Liu et al., ICRL for autonomous driving
- [4] Qiao et al., Multi-modal ICRL
- [7,8] HighD and inD datasets

## Code and Data Availability
The main code implementation, including training procedures, visualization tools, and datasets, is located in the `maximum-likelihood-constraint-inference` directory.

### Directory Structure
- `maximum-likelihood-constraint-inference/`: Contains the core implementation
  - Training code
  - Visualization tools
  - Dataset processing scripts

## Setup and Installation
- Python 3.x
- Required packages: numpy, torch, matplotlib, etc.
- Install dependencies: `pip install -r requirements.txt`

## Usage
- Prepare datasets in the required format
- Run training: `python train.py`
- Evaluate results: `python evaluate.py`
- Visualization scripts available in the visualization subdirectory

## Figures
For detailed figures and visualizations, please refer to the original report (EE364B_report.pdf) in the `maximum-likelihood-constraint-inference` directory.
