import numpy as np
import matplotlib.pyplot as plt
import json
import os


class MDPMetrics:
    def __init__(self, grid_mdp, training_epochs, output_dir="metrics_output"):
        """
        Initializes the metrics module for the given MDP.

        Args:
            grid_mdp: An instance of GridMDP containing the environment and policies.
            training_epochs: Number of training epochs for monitoring metrics.
            output_dir: Directory to save metrics files.
        """
        self.grid_mdp = grid_mdp
        self.training_epochs = training_epochs
        self.success_rate_collision = []
        self.success_rate_lane_change = []
        self.success_rate_combined = []
        self.average_rewards = []
        self.constraint_violation_rates = []
        self.output_dir = output_dir

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def calculate_metrics(self, demonstrations, constraints):
        """
        Calculates success rates, average reward, and constraint violation rate for a given set of demonstrations.

        Args:
            demonstrations: A list of trajectories (state-action sequences) from the MDP.
            constraints: A dictionary with constraints categorized as "collision" or "lane_change".

        Returns:
            metrics_dict: A dictionary with computed metrics.
        """
        successes_collision = 0
        successes_lane_change = 0
        total_violations = 0
        total_reward = 0
        num_trajectories = len(demonstrations[0])

        for state_seq, action_seq in zip(demonstrations[0], demonstrations[1]):
            # Check for collisions and lane changes
            collision_free = not any(state in constraints['collision'] for state in state_seq)
            lane_change_success = all(state in constraints['lane_change'] for state in state_seq)

            if collision_free:
                successes_collision += 1
            if lane_change_success:
                successes_lane_change += 1

            # Count total constraint violations
            violations = sum(
                state in constraints['collision'] or state not in constraints['lane_change'] for state in state_seq)
            total_violations += violations

            # Calculate reward for trajectory
            total_reward += sum(self.grid_mdp.rewards[state, action] for state, action in zip(state_seq, action_seq))

        # Calculate combined success rate
        combined_successes = successes_collision + successes_lane_change

        metrics_dict = {
            "success_rate_collision": successes_collision / num_trajectories,
            "success_rate_lane_change": successes_lane_change / num_trajectories,
            "success_rate_combined": combined_successes / (2 * num_trajectories),
            "average_reward": total_reward / num_trajectories,
            "constraint_violation_rate": total_violations / (num_trajectories * len(state_seq)),
        }

        return metrics_dict

    def record_training_metrics(self, epoch, demonstrations, constraints):
        """
        Records metrics during training for visualization and saving.

        Args:
            epoch: Current training epoch.
            demonstrations: A list of trajectories (state-action sequences) from the MDP.
            constraints: A dictionary with constraints categorized as "collision" or "lane_change".
        """
        metrics = self.calculate_metrics(demonstrations, constraints)

        self.success_rate_collision.append(metrics['success_rate_collision'])
        self.success_rate_lane_change.append(metrics['success_rate_lane_change'])
        self.success_rate_combined.append(metrics['success_rate_combined'])
        self.average_rewards.append(metrics['average_reward'])
        self.constraint_violation_rates.append(metrics['constraint_violation_rate'])

    def save_metrics_to_file(self):
        """
        Saves all recorded metrics to a JSON file after training.
        """
        metrics_data = {
            "success_rate_collision": self.success_rate_collision,
            "success_rate_lane_change": self.success_rate_lane_change,
            "success_rate_combined": self.success_rate_combined,
            "average_rewards": self.average_rewards,
            "constraint_violation_rates": self.constraint_violation_rates,
        }

        file_path = os.path.join(self.output_dir, "training_metrics.json")
        with open(file_path, "w") as f:
            json.dump(metrics_data, f, indent=4)

        print(f"Metrics saved to {file_path}")

    def plot_metrics(self):
        """
        Generates plots for Success Rate (Combined), Average Reward, and Constraint Violation Rate.
        """
        epochs = range(self.training_epochs)

        plt.figure(figsize=(18, 6))

        # Plot Success Rate (Combined)
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.success_rate_combined, label="Success Rate (Combined)", color='green')
        plt.xlabel("Epoch")
        plt.ylabel("Success Rate (Combined)")
        plt.title("Success Rate (Combined) During Training")
        plt.grid(True)
        plt.legend()

        # Plot Average Reward
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.average_rewards, label="Average Reward", color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.title("Average Reward During Training")
        plt.grid(True)
        plt.legend()

        # Plot Constraint Violation Rate
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.constraint_violation_rates, label="Constraint Violation Rate", color='red')
        plt.xlabel("Epoch")
        plt.ylabel("Constraint Violation Rate")
        plt.title("Constraint Violation Rate During Training")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

# Example usage:

# Initialize the MDPMetrics with a GridMDP instance and number of epochs
metrics = MDPMetrics(grid_mdp, training_epochs=50)

# During training, record metrics at each epoch:
for epoch in range(50):
    demonstrations = grid_mdp.produce_demonstrations(N=100, num_demos=50)
    constraints = {"collision": [list_of_collision_states], "lane_change": [list_of_lane_change_states]}
    metrics.record_training_metrics(epoch, demonstrations, constraints)

# Save metrics to file after training
metrics.save_metrics_to_file()

# Plot the recorded metrics after training
metrics.plot_metrics()
