#!/usr/bin/env python3
"""
Visualization script for RL with constraints results.
Creates a comprehensive figure showing:
- Left: Trajectory plots (x,y) for all generated trajectories
- Right: Three subfigures showing x/y position, velocity, and acceleration over time
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_generation_results(filepath):
    """
    Load generation results from pickle file.
    
    Args:
        filepath: Path to generation_results.pkl
    
    Returns:
        dict: Generation results data
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded generation results from {filepath}")
        print(f"  Number of trajectories: {len(data['trajectories'])}")
        print(f"  Track IDs: {data['track_ids']}")
        return data
    except Exception as e:
        print(f"❌ Error loading generation results: {e}")
        return None

def extract_trajectory_data(trajectories, real_trajectories=None, real_front_car_trajectories=None):
    """
    Extract trajectory data for visualization.
    
    Args:
        trajectories: List of trajectory states
        real_trajectories: List of real trajectory states (optional)
        real_front_car_trajectories: List of real front car trajectory states (optional)
    
    Returns:
        dict: Extracted trajectory data
    """
    trajectory_data = []
    
    for i, trajectory in enumerate(trajectories):
        # Extract data from each trajectory
        times = []
        x_positions = []
        y_positions = []
        velocities = []
        accelerations = []
        
        for j, state in enumerate(trajectory):
            # State format: [track_id, x, y, vx, vy, ax, ay, lon_vel, lat_vel, lon_acc, lat_acc, frame, ...]
            time = j * 0.4  # Assuming 0.4s timestep
            x = state[1]
            y = state[2]
            vx = state[3]
            vy = state[4]
            ax = state[5]
            ay = state[6]
            
            # Calculate magnitudes
            velocity_magnitude = np.sqrt(vx**2 + vy**2)
            acceleration_magnitude = np.sqrt(ax**2 + ay**2)
            
            times.append(time)
            x_positions.append(x)
            y_positions.append(y)
            velocities.append(velocity_magnitude)
            accelerations.append(acceleration_magnitude)
        
        traj_data = {
            'trajectory_id': i,
            'times': np.array(times),
            'x_positions': np.array(x_positions),
            'y_positions': np.array(y_positions),
            'velocities': np.array(velocities),
            'accelerations': np.array(accelerations),
            'track_id': trajectories[i][0][0] if trajectories[i] else None
        }
        
        # Add real trajectory data if available
        if real_trajectories and i < len(real_trajectories):
            real_traj = real_trajectories[i]
            real_times = []
            real_x_positions = []
            real_y_positions = []
            real_velocities = []
            real_accelerations = []
            
            for j, state in enumerate(real_traj):
                time = j * 0.4
                x = state[1]
                y = state[2]
                vx = state[3]
                vy = state[4]
                ax = state[5]
                ay = state[6]
                
                velocity_magnitude = np.sqrt(vx**2 + vy**2)
                acceleration_magnitude = np.sqrt(ax**2 + ay**2)
                
                real_times.append(time)
                real_x_positions.append(x)
                real_y_positions.append(y)
                real_velocities.append(velocity_magnitude)
                real_accelerations.append(acceleration_magnitude)
            
            traj_data.update({
                'real_times': np.array(real_times),
                'real_x_positions': np.array(real_x_positions),
                'real_y_positions': np.array(real_y_positions),
                'real_velocities': np.array(real_velocities),
                'real_accelerations': np.array(real_accelerations)
            })
        
        # Add real front car trajectory data if available
        if real_front_car_trajectories and i < len(real_front_car_trajectories):
            front_car_traj = real_front_car_trajectories[i]
            front_car_times = []
            front_car_x_positions = []
            front_car_y_positions = []
            front_car_velocities = []
            front_car_accelerations = []
            
            for j, state in enumerate(front_car_traj):
                time = j * 0.4
                x = state[1]
                y = state[2]
                vx = state[3]
                vy = state[4]
                ax = state[5]
                ay = state[6]
                
                velocity_magnitude = np.sqrt(vx**2 + vy**2)
                acceleration_magnitude = np.sqrt(ax**2 + ay**2)
                
                front_car_times.append(time)
                front_car_x_positions.append(x)
                front_car_y_positions.append(y)
                front_car_velocities.append(velocity_magnitude)
                front_car_accelerations.append(acceleration_magnitude)
            
            traj_data.update({
                'front_car_times': np.array(front_car_times),
                'front_car_x_positions': np.array(front_car_x_positions),
                'front_car_y_positions': np.array(front_car_y_positions),
                'front_car_velocities': np.array(front_car_velocities),
                'front_car_accelerations': np.array(front_car_accelerations)
            })
        
        trajectory_data.append(traj_data)
    
    return trajectory_data

def create_comprehensive_visualization(generation_results, output_path="./visualization/rl_constraints_visualization.png"):
    """
    Create comprehensive visualization of RL with constraints results.
    
    Args:
        generation_results: Dictionary containing generation results
        output_path: Path to save the visualization
    """
    if generation_results is None:
        print("❌ No generation results to visualize")
        return
    
    # Extract trajectory data
    trajectory_data = extract_trajectory_data(
        generation_results['trajectories'],
        generation_results.get('real_trajectories'),
        generation_results.get('real_front_car_trajectories')
    )
    
    if not trajectory_data:
        print("❌ No trajectory data to visualize")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Left side: Trajectory plots (x,y)
    ax_trajectories = fig.add_subplot(gs[:, 0:2])
    
    # Right side: Three subfigures
    ax_position = fig.add_subplot(gs[0, 2:4])
    ax_velocity = fig.add_subplot(gs[1, 2:4])
    ax_acceleration = fig.add_subplot(gs[2, 2:4])
    
    # Color palette for trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectory_data)))
    
    # Plot trajectories (left side)
    print("📊 Creating trajectory plots...")
    for i, traj in enumerate(trajectory_data):
        color = colors[i]
        track_id = traj['track_id']
        
        # Plot trajectory
        ax_trajectories.plot(traj['x_positions'], traj['y_positions'], 
                           'o-', color=color, linewidth=2, markersize=4, 
                           label=f'Track {track_id}', alpha=0.8)
        
        # Mark start and end points
        ax_trajectories.plot(traj['x_positions'][0], traj['y_positions'][0], 
                           'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=2)
        ax_trajectories.plot(traj['x_positions'][-1], traj['y_positions'][-1], 
                           's', color=color, markersize=8, markeredgecolor='black', markeredgewidth=2)
    
    # Add goal points if available
    if 'goals' in generation_results:
        goals = generation_results['goals']
        goal_x = [goal[0] for goal in goals]
        goal_y = [goal[1] for goal in goals]
        ax_trajectories.scatter(goal_x, goal_y, c='red', s=100, marker='*', 
                              label='Goals', zorder=10, edgecolors='black', linewidth=1)
    
    ax_trajectories.set_title('Generated Trajectories (x,y)', fontsize=16, fontweight='bold')
    ax_trajectories.set_xlabel('X Position', fontsize=12)
    ax_trajectories.set_ylabel('Y Position', fontsize=12)
    ax_trajectories.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax_trajectories.grid(True, alpha=0.3)
    ax_trajectories.set_aspect('equal')
    
    # Plot position over time (top right)
    print("📊 Creating position plots...")
    for i, traj in enumerate(trajectory_data):
        color = colors[i]
        ax_position.plot(traj['times'], traj['x_positions'], 
                        'o-', color=color, linewidth=2, markersize=3, 
                        label=f'X - Track {traj["track_id"]}', alpha=0.7)
        ax_position.plot(traj['times'], traj['y_positions'], 
                        's--', color=color, linewidth=2, markersize=3, 
                        label=f'Y - Track {traj["track_id"]}', alpha=0.7)
    
    ax_position.set_title('Position Over Time', fontsize=14, fontweight='bold')
    ax_position.set_xlabel('Time (s)', fontsize=12)
    ax_position.set_ylabel('Position', fontsize=12)
    ax_position.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax_position.grid(True, alpha=0.3)
    
    # Plot velocity over time (middle right)
    print("📊 Creating velocity plots...")
    for i, traj in enumerate(trajectory_data):
        color = colors[i]
        ax_velocity.plot(traj['times'], traj['velocities'], 
                        'o-', color=color, linewidth=2, markersize=3, 
                        label=f'Track {traj["track_id"]}', alpha=0.7)
    
    ax_velocity.set_title('Velocity Magnitude Over Time', fontsize=14, fontweight='bold')
    ax_velocity.set_xlabel('Time (s)', fontsize=12)
    ax_velocity.set_ylabel('Velocity (m/s)', fontsize=12)
    ax_velocity.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax_velocity.grid(True, alpha=0.3)
    
    # Add velocity constraint line
    max_velocity = 30.0  # From the RL code
    ax_velocity.axhline(y=max_velocity, color='red', linestyle='--', alpha=0.7, 
                       label=f'Max Velocity ({max_velocity} m/s)')
    
    # Plot acceleration over time (bottom right)
    print("📊 Creating acceleration plots...")
    for i, traj in enumerate(trajectory_data):
        color = colors[i]
        ax_acceleration.plot(traj['times'], traj['accelerations'], 
                           'o-', color=color, linewidth=2, markersize=3, 
                           label=f'Track {traj["track_id"]}', alpha=0.7)
    
    ax_acceleration.set_title('Acceleration Magnitude Over Time', fontsize=14, fontweight='bold')
    ax_acceleration.set_xlabel('Time (s)', fontsize=12)
    ax_acceleration.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax_acceleration.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax_acceleration.grid(True, alpha=0.3)
    
    # Add acceleration constraint line
    max_acceleration = 5.0  # From the RL code
    ax_acceleration.axhline(y=max_acceleration, color='red', linestyle='--', alpha=0.7, 
                           label=f'Max Acceleration ({max_acceleration} m/s²)')
    
    # Add overall title
    fig.suptitle('RL with Constraints: Generated Trajectories Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add statistics text
    rewards = generation_results.get('rewards', [])
    if rewards:
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        stats_text = f'Average Reward: {avg_reward:.2f} ± {std_reward:.2f}'
        fig.text(0.02, 0.02, stats_text, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    # Save the figure
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {output_path}")
    
    return trajectory_data

def create_individual_trajectory_plot(trajectory_data, generation_results, output_dir="./visualization"):
    """
    Create individual plots for each trajectory.
    
    Args:
        trajectory_data: Extracted trajectory data
        generation_results: Generation results
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Creating individual trajectory plots...")
    
    for i, traj in enumerate(trajectory_data):
        # Create figure for this trajectory
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Left side: Trajectory plot (x,y)
        ax_trajectory = fig.add_subplot(gs[:, 0:2])
        
        # Right side: Three subfigures
        ax_position = fig.add_subplot(gs[0, 2:4])
        ax_velocity = fig.add_subplot(gs[1, 2:4])
        ax_acceleration = fig.add_subplot(gs[2, 2:4])
        
        # Color for this trajectory
        color = plt.cm.tab10(i / len(trajectory_data))
        track_id = traj['track_id']
        
        # Plot planned trajectory (left side)
        ax_trajectory.plot(traj['x_positions'], traj['y_positions'], 
                          'o-', color=color, linewidth=3, markersize=6, 
                          label=f'Planned - Track {track_id}', alpha=0.8)
        
        # Mark start and end points for planned trajectory
        ax_trajectory.plot(traj['x_positions'][0], traj['y_positions'][0], 
                          'o', color=color, markersize=12, markeredgecolor='black', 
                          markeredgewidth=2, label='Planned Start')
        ax_trajectory.plot(traj['x_positions'][-1], traj['y_positions'][-1], 
                          's', color=color, markersize=12, markeredgecolor='black', 
                          markeredgewidth=2, label='Planned End')
        
        # Plot real trajectory if available
        if 'real_x_positions' in traj and 'real_y_positions' in traj:
            ax_trajectory.plot(traj['real_x_positions'], traj['real_y_positions'], 
                              '^-', color='red', linewidth=2, markersize=4, 
                              label=f'Real - Track {track_id}', alpha=0.8)
            ax_trajectory.plot(traj['real_x_positions'][0], traj['real_y_positions'][0], 
                              '^', color='red', markersize=10, markeredgecolor='black', 
                              markeredgewidth=2, label='Real Start')
            ax_trajectory.plot(traj['real_x_positions'][-1], traj['real_y_positions'][-1], 
                              'D', color='red', markersize=10, markeredgecolor='black', 
                              markeredgewidth=2, label='Real End')
        
        # Plot front car trajectory if available
        if 'front_car_x_positions' in traj and 'front_car_y_positions' in traj:
            ax_trajectory.plot(traj['front_car_x_positions'], traj['front_car_y_positions'], 
                              's--', color='blue', linewidth=2, markersize=4, 
                              label=f'Front Car', alpha=0.6)
        
        # Add goal point if available
        if 'goals' in generation_results and i < len(generation_results['goals']):
            goal = generation_results['goals'][i]
            ax_trajectory.scatter(goal[0], goal[1], c='red', s=150, marker='*', 
                                label='Goal', zorder=10, edgecolors='black', linewidth=1)
        
        ax_trajectory.set_title(f'Trajectory {i+1}: Track {track_id}', fontsize=16, fontweight='bold')
        ax_trajectory.set_xlabel('X Position', fontsize=12)
        ax_trajectory.set_ylabel('Y Position', fontsize=12)
        ax_trajectory.legend(fontsize=10)
        ax_trajectory.grid(True, alpha=0.3)
        ax_trajectory.set_aspect('equal')
        
        # Plot position over time (top right)
        ax_position.plot(traj['times'], traj['x_positions'], 
                        'o-', color=color, linewidth=2, markersize=4, 
                        label='Planned X', alpha=0.8)
        ax_position.plot(traj['times'], traj['y_positions'], 
                        's--', color=color, linewidth=2, markersize=4, 
                        label='Planned Y', alpha=0.8)
        
        # Plot real position if available
        if 'real_times' in traj and 'real_x_positions' in traj:
            ax_position.plot(traj['real_times'], traj['real_x_positions'], 
                            '^-', color='red', linewidth=2, markersize=3, 
                            label='Real X', alpha=0.7)
            ax_position.plot(traj['real_times'], traj['real_y_positions'], 
                            'D--', color='red', linewidth=2, markersize=3, 
                            label='Real Y', alpha=0.7)
        
        ax_position.set_title('Position Over Time', fontsize=14, fontweight='bold')
        ax_position.set_xlabel('Time (s)', fontsize=12)
        ax_position.set_ylabel('Position', fontsize=12)
        ax_position.legend(fontsize=10)
        ax_position.grid(True, alpha=0.3)
        
        # Plot velocity over time (middle right)
        ax_velocity.plot(traj['times'], traj['velocities'], 
                        'o-', color=color, linewidth=2, markersize=4, 
                        label='Planned Velocity', alpha=0.8)
        
        # Plot real velocity if available
        if 'real_times' in traj and 'real_velocities' in traj:
            ax_velocity.plot(traj['real_times'], traj['real_velocities'], 
                            '^-', color='red', linewidth=2, markersize=3, 
                            label='Real Velocity', alpha=0.7)
        
        # Add velocity constraint line
        max_velocity = 30.0
        ax_velocity.axhline(y=max_velocity, color='red', linestyle='--', alpha=0.7, 
                           label=f'Max Velocity ({max_velocity} m/s)')
        
        ax_velocity.set_title('Velocity Magnitude Over Time', fontsize=14, fontweight='bold')
        ax_velocity.set_xlabel('Time (s)', fontsize=12)
        ax_velocity.set_ylabel('Velocity (m/s)', fontsize=12)
        ax_velocity.legend(fontsize=10)
        ax_velocity.grid(True, alpha=0.3)
        
        # Plot acceleration over time (bottom right)
        ax_acceleration.plot(traj['times'], traj['accelerations'], 
                           'o-', color=color, linewidth=2, markersize=4, 
                           label='Planned Acceleration', alpha=0.8)
        
        # Plot real acceleration if available
        if 'real_times' in traj and 'real_accelerations' in traj:
            ax_acceleration.plot(traj['real_times'], traj['real_accelerations'], 
                               '^-', color='red', linewidth=2, markersize=3, 
                               label='Real Acceleration', alpha=0.7)
        
        # Add acceleration constraint line
        max_acceleration = 5.0
        ax_acceleration.axhline(y=max_acceleration, color='red', linestyle='--', alpha=0.7, 
                               label=f'Max Acceleration ({max_acceleration} m/s²)')
        
        ax_acceleration.set_title('Acceleration Magnitude Over Time', fontsize=14, fontweight='bold')
        ax_acceleration.set_xlabel('Time (s)', fontsize=12)
        ax_acceleration.set_ylabel('Acceleration (m/s²)', fontsize=12)
        ax_acceleration.legend(fontsize=10)
        ax_acceleration.grid(True, alpha=0.3)
        
        # Add trajectory statistics
        reward = generation_results.get('rewards', [0])[i] if i < len(generation_results.get('rewards', [])) else 0
        final_velocity = traj['velocities'][-1]
        final_acceleration = traj['accelerations'][-1]
        distance_traveled = np.sqrt((traj['x_positions'][-1] - traj['x_positions'][0])**2 + 
                                   (traj['y_positions'][-1] - traj['y_positions'][0])**2)
        
        stats_text = f'Reward: {reward:.2f}\nFinal Velocity: {final_velocity:.2f} m/s\nFinal Acceleration: {final_acceleration:.2f} m/s²\nDistance: {distance_traveled:.2f} m'
        fig.text(0.02, 0.02, stats_text, fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Save individual trajectory plot
        trajectory_filename = f"trajectory_{i+1:02d}_track_{track_id}.png"
        plt.savefig(output_dir / trajectory_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved trajectory {i+1}/10: {trajectory_filename}")
    
    print(f"✅ All individual trajectory plots saved to: {output_dir}")

def create_additional_analysis_plots(trajectory_data, generation_results, output_dir="./visualization"):
    """
    Create additional analysis plots.
    
    Args:
        trajectory_data: Extracted trajectory data
        generation_results: Generation results
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Reward distribution
    plt.figure(figsize=(10, 6))
    rewards = generation_results.get('rewards', [])
    if rewards:
        plt.hist(rewards, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        plt.title('Distribution of Trajectory Rewards', fontsize=14, fontweight='bold')
        plt.xlabel('Reward', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "reward_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Trajectory length analysis
    plt.figure(figsize=(10, 6))
    lengths = [len(traj['times']) for traj in trajectory_data]
    plt.hist(lengths, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
    plt.title('Distribution of Trajectory Lengths', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Steps', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "trajectory_length_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Constraint violation analysis
    plt.figure(figsize=(12, 8))
    
    # Count violations for each trajectory
    velocity_violations = []
    acceleration_violations = []
    
    for traj in trajectory_data:
        # Count velocity violations
        vel_violations = np.sum(traj['velocities'] > 30.0)
        velocity_violations.append(vel_violations)
        
        # Count acceleration violations
        acc_violations = np.sum(traj['accelerations'] > 5.0)
        acceleration_violations.append(acc_violations)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Velocity violations
    ax1.bar(range(len(velocity_violations)), velocity_violations, alpha=0.7, color='orange')
    ax1.set_title('Velocity Constraint Violations', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Trajectory ID', fontsize=10)
    ax1.set_ylabel('Number of Violations', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Acceleration violations
    ax2.bar(range(len(acceleration_violations)), acceleration_violations, alpha=0.7, color='purple')
    ax2.set_title('Acceleration Constraint Violations', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trajectory ID', fontsize=10)
    ax2.set_ylabel('Number of Violations', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "constraint_violations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Additional analysis plots saved to: {output_dir}")

def create_summary_report(trajectory_data, generation_results, output_dir="./visualization"):
    """
    Create a summary report of the analysis.
    
    Args:
        trajectory_data: Extracted trajectory data
        generation_results: Generation results
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics
    rewards = generation_results.get('rewards', [])
    lengths = [len(traj['times']) for traj in trajectory_data]
    
    # Constraint violations
    total_velocity_violations = 0
    total_acceleration_violations = 0
    total_steps = 0
    
    for traj in trajectory_data:
        vel_violations = np.sum(traj['velocities'] > 30.0)
        acc_violations = np.sum(traj['accelerations'] > 5.0)
        total_velocity_violations += vel_violations
        total_acceleration_violations += acc_violations
        total_steps += len(traj['times'])
    
    # Create report
    report = f"""
RL with Constraints - Generation Results Summary
================================================

Dataset Information:
- Total trajectories generated: {len(trajectory_data)}
- Track IDs used: {generation_results.get('track_ids', [])}

Performance Metrics:
- Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}
- Min reward: {np.min(rewards):.2f}
- Max reward: {np.max(rewards):.2f}
- Average trajectory length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} steps

Constraint Satisfaction:
- Total steps: {total_steps}
- Velocity violations: {total_velocity_violations} ({total_velocity_violations/total_steps*100:.2f}%)
- Acceleration violations: {total_acceleration_violations} ({total_acceleration_violations/total_steps*100:.2f}%)

Trajectory Analysis:
- Average final velocity: {np.mean([traj['velocities'][-1] for traj in trajectory_data]):.2f} m/s
- Average final acceleration: {np.mean([traj['accelerations'][-1] for traj in trajectory_data]):.2f} m/s²
- Average distance traveled: {np.mean([np.sqrt((traj['x_positions'][-1] - traj['x_positions'][0])**2 + (traj['y_positions'][-1] - traj['y_positions'][0])**2) for traj in trajectory_data]):.2f} m

Goals Information:
- Number of goals: {len(generation_results.get('goals', []))}
- Goals: {generation_results.get('goals', [])}
"""
    
    # Save report
    with open(output_dir / "generation_summary_report.txt", 'w') as f:
        f.write(report)
    
    print(f"✅ Summary report saved to: {output_dir / 'generation_summary_report.txt'}")
    print("\n" + report)

def main():
    """
    Main function to run the visualization.
    """
    print("🎨 RL with Constraints Visualization")
    print("=" * 50)
    
    # Load generation results
    generation_results = load_generation_results("./trajectory_data/generation_results.pkl")
    
    if generation_results is None:
        print("❌ Could not load generation results. Please run the RL training first.")
        return
    
    # Create comprehensive visualization
    print("\n📊 Creating comprehensive visualization...")
    trajectory_data = create_comprehensive_visualization(
        generation_results, 
        output_path="./visualization/rl_constraints_visualization.png"
    )
    
    # Create individual trajectory plots
    print("\n📊 Creating individual trajectory plots...")
    create_individual_trajectory_plot(trajectory_data, generation_results)
    
    # Create additional analysis plots
    print("\n📊 Creating additional analysis plots...")
    create_additional_analysis_plots(trajectory_data, generation_results)
    
    # Create summary report
    print("\n📊 Creating summary report...")
    create_summary_report(trajectory_data, generation_results)
    
    print("\n✅ Visualization complete!")
    print("📁 Check the './visualization/' directory for all generated plots and reports.")

if __name__ == "__main__":
    main() 