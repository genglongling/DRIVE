"""
Visualize Integrated Results
===========================

This module provides comprehensive visualization for the integrated methods comparison
results, including trajectory comparison, performance metrics, and statistical analysis.

This is the FULL version that uses matplotlib and seaborn for advanced visualizations.

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class IntegratedResultsVisualizer:
    """
    Comprehensive visualizer for integrated methods comparison results.
    This version uses matplotlib and seaborn for advanced visualizations.
    """
    
    def __init__(self, data_dir: str = "integrated_results", output_dir: str = "visualization_output"):
        """
        Initialize the visualizer.
        
        Args:
            data_dir: Directory containing the integrated results
            output_dir: Directory to save visualizations
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.trajectory_data = None
        self.performance_data = None
        self.load_data()
        
        # Color scheme for methods
        self.method_colors = {
            'beam_search': '#1f77b4',
            'mdp_icl': '#ff7f0e', 
            'icl_convex': '#2ca02c',
            'cpo': '#d62728',
            'dpo': '#9467bd'
        }
        
        # Method display names
        self.method_names = {
            'beam_search': 'Beam Search',
            'mdp_icl': 'MDP-ICL',
            'icl_convex': 'ICL + Convex',
            'cpo': 'CPO',
            'dpo': 'DPO'
        }
        
        print("🎨 Integrated Results Visualizer Initialized (FULL VERSION)")
        print(f"📁 Data Directory: {data_dir}")
        print(f"📁 Output Directory: {output_dir}")
    
    def load_data(self):
        """Load trajectory and performance data."""
        try:
            # Load trajectory data
            trajectory_file = f"{self.data_dir}/integrated_trajectory_data.csv"
            if os.path.exists(trajectory_file):
                self.trajectory_data = pd.read_csv(trajectory_file)
                print(f"✅ Loaded trajectory data: {len(self.trajectory_data)} entries")
            else:
                print(f"⚠️  Trajectory data not found: {trajectory_file}")
            
            # Load performance data
            performance_file = f"{self.data_dir}/integrated_results.csv"
            if os.path.exists(performance_file):
                self.performance_data = pd.read_csv(performance_file, index_col=0)
                print(f"✅ Loaded performance data: {len(self.performance_data)} methods")
            else:
                print(f"⚠️  Performance data not found: {performance_file}")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def plot_trajectory_comparison(self, save_path: str = "trajectory_comparison.png"):
        """Plot trajectory comparison for all methods."""
        if self.trajectory_data is None:
            print("❌ No trajectory data available")
            return
        
        print("📊 Creating trajectory comparison plot...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Trajectory Comparison Across Methods', fontsize=16, fontweight='bold')
        
        methods = self.trajectory_data['method_name'].unique()
        
        for idx, method in enumerate(methods):
            if idx >= 6:  # Limit to 6 subplots
                break
                
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            method_data = self.trajectory_data[self.trajectory_data['method_name'] == method]
            
            # Plot planned trajectory
            ax.plot(method_data['planned_ego_x'], method_data['planned_ego_y'], 
                   'o-', color=self.method_colors.get(method, '#666666'), 
                   linewidth=2, markersize=4, label='Planned')
            
            # Plot real trajectory
            ax.plot(method_data['real_ego_x'], method_data['real_ego_y'], 
                   's--', color='red', linewidth=2, markersize=4, label='Real')
            
            # Add start and end points
            if len(method_data) > 0:
                ax.plot(method_data['planned_ego_x'].iloc[0], method_data['planned_ego_y'].iloc[0], 
                       'go', markersize=8, label='Start')
                ax.plot(method_data['planned_ego_x'].iloc[-1], method_data['planned_ego_y'].iloc[-1], 
                       'ro', markersize=8, label='End')
            
            ax.set_title(f'{self.method_names.get(method, method)}', fontweight='bold')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # Remove empty subplots
        for idx in range(len(methods), 6):
            row, col = idx // 3, idx % 3
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_path}", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Trajectory comparison saved to {save_path}")
    
    def plot_velocity_comparison(self, save_path: str = "velocity_comparison.png"):
        """Plot velocity magnitude over time for all methods."""
        if self.trajectory_data is None:
            print("❌ No trajectory data available")
            return
        
        print("📊 Creating velocity comparison plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for method in self.trajectory_data['method_name'].unique():
            method_data = self.trajectory_data[self.trajectory_data['method_name'] == method]
            
            # Calculate velocity magnitude
            planned_velocity = np.sqrt(method_data['planned_ego_vx']**2 + method_data['planned_ego_vy']**2)
            real_velocity = np.sqrt(method_data['real_ego_vx']**2 + method_data['real_ego_vy']**2)
            
            ax.plot(method_data['step'], planned_velocity, 
                   'o-', color=self.method_colors.get(method, '#666666'), 
                   linewidth=2, markersize=4, 
                   label=f'{self.method_names.get(method, method)} (Planned)')
            ax.plot(method_data['step'], real_velocity, 
                   's--', color=self.method_colors.get(method, '#666666'), 
                   linewidth=2, markersize=4, alpha=0.7,
                   label=f'{self.method_names.get(method, method)} (Real)')
        
        ax.set_title('Velocity Magnitude Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Velocity Magnitude (m/s)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_path}", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Velocity comparison saved to {save_path}")
    
    def plot_acceleration_comparison(self, save_path: str = "acceleration_comparison.png"):
        """Plot acceleration magnitude over time for all methods."""
        if self.trajectory_data is None:
            print("❌ No trajectory data available")
            return
        
        print("📊 Creating acceleration comparison plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for method in self.trajectory_data['method_name'].unique():
            method_data = self.trajectory_data[self.trajectory_data['method_name'] == method]
            
            # Calculate acceleration magnitude
            planned_acceleration = np.sqrt(method_data['planned_ego_ax']**2 + method_data['planned_ego_ay']**2)
            real_acceleration = np.sqrt(method_data['real_ego_ax']**2 + method_data['real_ego_ay']**2)
            
            ax.plot(method_data['step'], planned_acceleration, 
                   'o-', color=self.method_colors.get(method, '#666666'), 
                   linewidth=2, markersize=4, 
                   label=f'{self.method_names.get(method, method)} (Planned)')
            ax.plot(method_data['step'], real_acceleration, 
                   's--', color=self.method_colors.get(method, '#666666'), 
                   linewidth=2, markersize=4, alpha=0.7,
                   label=f'{self.method_names.get(method, method)} (Real)')
        
        ax.set_title('Acceleration Magnitude Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Acceleration Magnitude (m/s²)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_path}", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Acceleration comparison saved to {save_path}")
    
    def plot_performance_metrics(self, save_path: str = "performance_metrics.png"):
        """Plot performance metrics comparison."""
        if self.performance_data is None:
            print("❌ No performance data available")
            return
        
        print("📊 Creating performance metrics plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Training Time
        ax1 = axes[0, 0]
        training_times = self.performance_data['training_time']
        colors = [self.method_colors.get(method, '#666666') for method in training_times.index]
        bars1 = ax1.bar(range(len(training_times)), training_times.values, color=colors, alpha=0.8)
        ax1.set_title('Training Time', fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xticks(range(len(training_times)))
        ax1.set_xticklabels([self.method_names.get(method, method) for method in training_times.index], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, training_times.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}s', ha='center', va='bottom', fontsize=9)
        
        # Peak Memory Usage
        ax2 = axes[0, 1]
        memory_usage = self.performance_data['peak_memory_mb']
        bars2 = ax2.bar(range(len(memory_usage)), memory_usage.values, color=colors, alpha=0.8)
        ax2.set_title('Peak Memory Usage', fontweight='bold')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_xticks(range(len(memory_usage)))
        ax2.set_xticklabels([self.method_names.get(method, method) for method in memory_usage.index], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, memory_usage.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}MB', ha='center', va='bottom', fontsize=9)
        
        # CPU Time
        ax3 = axes[1, 0]
        cpu_times = self.performance_data['cpu_time']
        bars3 = ax3.bar(range(len(cpu_times)), cpu_times.values, color=colors, alpha=0.8)
        ax3.set_title('CPU Time', fontweight='bold')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xticks(range(len(cpu_times)))
        ax3.set_xticklabels([self.method_names.get(method, method) for method in cpu_times.index], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars3, cpu_times.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                    f'{value:.3f}s', ha='center', va='bottom', fontsize=9)
        
        # Memory Efficiency
        ax4 = axes[1, 1]
        memory_efficiency = self.performance_data['memory_efficiency']
        bars4 = ax4.bar(range(len(memory_efficiency)), memory_efficiency.values, color=colors, alpha=0.8)
        ax4.set_title('Memory Efficiency', fontweight='bold')
        ax4.set_ylabel('Efficiency (CPU Time / Memory)')
        ax4.set_xticks(range(len(memory_efficiency)))
        ax4.set_xticklabels([self.method_names.get(method, method) for method in memory_efficiency.index], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars4, memory_efficiency.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_path}", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Performance metrics saved to {save_path}")
    
    def plot_method_comparison_summary(self, save_path: str = "method_comparison_summary.png"):
        """Create a comprehensive summary plot."""
        if self.trajectory_data is None or self.performance_data is None:
            print("❌ Insufficient data for summary plot")
            return
        
        print("📊 Creating comprehensive summary plot...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Trajectory comparison (top left)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        for method in self.trajectory_data['method_name'].unique():
            method_data = self.trajectory_data[self.trajectory_data['method_name'] == method]
            ax1.plot(method_data['planned_ego_x'], method_data['planned_ego_y'], 
                    'o-', color=self.method_colors.get(method, '#666666'), 
                    linewidth=2, markersize=4, 
                    label=self.method_names.get(method, method))
        ax1.set_title('Trajectory Comparison', fontweight='bold')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. Performance radar chart (top right)
        ax2 = fig.add_subplot(gs[0:2, 2:4], projection='polar')
        
        # Normalize metrics for radar chart
        metrics = ['training_time', 'peak_memory_mb', 'cpu_time', 'memory_efficiency']
        normalized_data = {}
        
        for metric in metrics:
            values = self.performance_data[metric]
            # Invert metrics where lower is better
            if metric in ['training_time', 'peak_memory_mb', 'cpu_time']:
                normalized_data[metric] = 1 - (values - values.min()) / (values.max() - values.min())
            else:
                normalized_data[metric] = (values - values.min()) / (values.max() - values.min())
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for method in self.performance_data.index:
            values = [normalized_data[metric][method] for metric in metrics]
            values += values[:1]  # Complete the circle
            ax2.plot(angles, values, 'o-', linewidth=2, 
                    label=self.method_names.get(method, method),
                    color=self.method_colors.get(method, '#666666'))
            ax2.fill(angles, values, alpha=0.1, color=self.method_colors.get(method, '#666666'))
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(['Training\nTime', 'Memory\nUsage', 'CPU\nTime', 'Memory\nEfficiency'])
        ax2.set_ylim(0, 1)
        ax2.set_title('Performance Radar Chart', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 3. Velocity comparison (bottom left)
        ax3 = fig.add_subplot(gs[2, 0:2])
        for method in self.trajectory_data['method_name'].unique():
            method_data = self.trajectory_data[self.trajectory_data['method_name'] == method]
            velocity = np.sqrt(method_data['planned_ego_vx']**2 + method_data['planned_ego_vy']**2)
            ax3.plot(method_data['step'], velocity, 
                    'o-', color=self.method_colors.get(method, '#666666'), 
                    linewidth=2, markersize=4, 
                    label=self.method_names.get(method, method))
        ax3.set_title('Velocity Over Time', fontweight='bold')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance bar chart (bottom right)
        ax4 = fig.add_subplot(gs[2, 2:4])
        training_times = self.performance_data['training_time']
        colors = [self.method_colors.get(method, '#666666') for method in training_times.index]
        bars = ax4.bar(range(len(training_times)), training_times.values, color=colors, alpha=0.8)
        ax4.set_title('Training Time Comparison', fontweight='bold')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_xticks(range(len(training_times)))
        ax4.set_xticklabels([self.method_names.get(method, method) for method in training_times.index], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, training_times.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}s', ha='center', va='bottom', fontsize=9)
        
        # 5. Statistics table (bottom)
        ax5 = fig.add_subplot(gs[3, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create summary statistics
        stats_data = []
        for method in self.performance_data.index:
            method_data = self.trajectory_data[self.trajectory_data['method_name'] == method]
            stats_data.append([
                self.method_names.get(method, method),
                f"{self.performance_data.loc[method, 'training_time']:.3f}s",
                f"{self.performance_data.loc[method, 'peak_memory_mb']:.1f}MB",
                f"{self.performance_data.loc[method, 'cpu_time']:.3f}s",
                f"{len(method_data)} steps"
            ])
        
        table = ax5.table(cellText=stats_data,
                         colLabels=['Method', 'Training Time', 'Peak Memory', 'CPU Time', 'Trajectory Steps'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color table headers
        for i in range(len(stats_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.savefig(f"{self.output_dir}/{save_path}", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive summary saved to {save_path}")
    
    def create_text_summary(self):
        """Create a text-based summary of the results."""
        print("📊 Creating text summary...")
        
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("INTEGRATED METHODS COMPARISON - TEXT SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        if self.trajectory_data is not None:
            summary_lines.append("TRAJECTORY DATA SUMMARY:")
            summary_lines.append("-" * 40)
            summary_lines.append(f"Total trajectory points: {len(self.trajectory_data)}")
            
            # Group by method
            method_counts = self.trajectory_data['method_name'].value_counts()
            for method, count in method_counts.items():
                name = self.method_names.get(method, method)
                summary_lines.append(f"{name}: {count} trajectory points")
            summary_lines.append("")
        
        if self.performance_data is not None:
            summary_lines.append("PERFORMANCE METRICS SUMMARY:")
            summary_lines.append("-" * 40)
            
            # Sort by training time
            sorted_methods = self.performance_data.sort_values('training_time')
            
            for method in sorted_methods.index:
                name = self.method_names.get(method, method)
                metrics = sorted_methods.loc[method]
                summary_lines.append(f"{name.upper()}:")
                summary_lines.append(f"  Training Time: {metrics['training_time']:.3f}s")
                summary_lines.append(f"  Peak Memory: {metrics['peak_memory_mb']:.1f}MB")
                summary_lines.append(f"  CPU Time: {metrics['cpu_time']:.3f}s")
                summary_lines.append(f"  Memory Efficiency: {metrics['memory_efficiency']:.3f}")
                summary_lines.append("")
            
            # Find best performing method
            fastest_method = self.performance_data['training_time'].idxmin()
            most_efficient = self.performance_data['peak_memory_mb'].idxmin()
            
            summary_lines.append("PERFORMANCE RANKINGS:")
            summary_lines.append("-" * 40)
            summary_lines.append(f"Fastest Training: {self.method_names.get(fastest_method, fastest_method)} ({self.performance_data.loc[fastest_method, 'training_time']:.3f}s)")
            summary_lines.append(f"Most Memory Efficient: {self.method_names.get(most_efficient, most_efficient)} ({self.performance_data.loc[most_efficient, 'peak_memory_mb']:.1f}MB)")
            summary_lines.append("")
        
        # Save summary
        summary_path = f"{self.output_dir}/text_summary.txt"
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"✅ Text summary saved to {summary_path}")
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("TEXT SUMMARY")
        print("=" * 80)
        for line in summary_lines[:30]:  # Print first 30 lines
            print(line)
        if len(summary_lines) > 30:
            print("... (see full summary in text_summary.txt)")
    
    def create_csv_summary(self):
        """Create CSV summary files."""
        print("📊 Creating CSV summaries...")
        
        if self.trajectory_data is not None:
            # Save trajectory data as CSV
            trajectory_csv_path = f"{self.output_dir}/trajectory_summary.csv"
            self.trajectory_data.to_csv(trajectory_csv_path, index=False)
            print(f"✅ Trajectory summary saved to {trajectory_csv_path}")
        
        if self.performance_data is not None:
            # Create performance summary with display names
            performance_summary = self.performance_data.copy()
            performance_summary['display_name'] = [self.method_names.get(method, method) for method in performance_summary.index]
            performance_summary = performance_summary.reset_index()
            performance_summary = performance_summary.rename(columns={'index': 'method'})
            
            # Reorder columns
            cols = ['method', 'display_name', 'training_time', 'peak_memory_mb', 'cpu_time', 'memory_efficiency']
            performance_summary = performance_summary[cols]
            
            performance_csv_path = f"{self.output_dir}/performance_summary.csv"
            performance_summary.to_csv(performance_csv_path, index=False)
            print(f"✅ Performance summary saved to {performance_csv_path}")
    
    def create_statistics_report(self):
        """Generate comprehensive statistics report."""
        print("📋 Generating statistics report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("INTEGRATED METHODS COMPARISON - STATISTICS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if self.trajectory_data is not None:
            report_lines.append("TRAJECTORY DATA STATISTICS:")
            report_lines.append("-" * 40)
            report_lines.append(f"Total trajectory points: {len(self.trajectory_data)}")
            
            # Group by method
            for method in self.trajectory_data['method_name'].unique():
                method_data = self.trajectory_data[self.trajectory_data['method_name'] == method]
                name = self.method_names.get(method, method)
                
                # Calculate statistics
                planned_velocities = np.sqrt(method_data['planned_ego_vx']**2 + method_data['planned_ego_vy']**2)
                planned_accelerations = np.sqrt(method_data['planned_ego_ax']**2 + method_data['planned_ego_ay']**2)
                distances = method_data['closest_car_distance']
                
                report_lines.append(f"\n{name.upper()}:")
                report_lines.append(f"  Trajectory points: {len(method_data)}")
                report_lines.append(f"  Average velocity: {planned_velocities.mean():.3f} ± {planned_velocities.std():.3f}")
                report_lines.append(f"  Max velocity: {planned_velocities.max():.3f}")
                report_lines.append(f"  Average acceleration: {planned_accelerations.mean():.3f} ± {planned_accelerations.std():.3f}")
                report_lines.append(f"  Max acceleration: {planned_accelerations.max():.3f}")
                report_lines.append(f"  Average safety distance: {distances.mean():.3f} ± {distances.std():.3f}")
                report_lines.append(f"  Min safety distance: {distances.min():.3f}")
        
        if self.performance_data is not None:
            report_lines.append("\n\nPERFORMANCE METRICS:")
            report_lines.append("-" * 40)
            
            # Calculate performance statistics
            training_times = self.performance_data['training_time']
            memory_usage = self.performance_data['peak_memory_mb']
            cpu_times = self.performance_data['cpu_time']
            
            report_lines.append(f"Training Time Statistics:")
            report_lines.append(f"  Average: {training_times.mean():.3f}s ± {training_times.std():.3f}s")
            report_lines.append(f"  Min: {training_times.min():.3f}s")
            report_lines.append(f"  Max: {training_times.max():.3f}s")
            report_lines.append(f"  Range: {training_times.max() - training_times.min():.3f}s")
            
            report_lines.append(f"\nMemory Usage Statistics:")
            report_lines.append(f"  Average: {memory_usage.mean():.1f}MB ± {memory_usage.std():.1f}MB")
            report_lines.append(f"  Min: {memory_usage.min():.1f}MB")
            report_lines.append(f"  Max: {memory_usage.max():.1f}MB")
            
            report_lines.append(f"\nCPU Time Statistics:")
            report_lines.append(f"  Average: {cpu_times.mean():.3f}s ± {cpu_times.std():.3f}s")
            report_lines.append(f"  Min: {cpu_times.min():.3f}s")
            report_lines.append(f"  Max: {cpu_times.max():.3f}s")
            
            # Best performing methods
            fastest_method = training_times.idxmin()
            most_efficient = memory_usage.idxmin()
            
            report_lines.append(f"\nPERFORMANCE RANKINGS:")
            report_lines.append(f"  Fastest: {self.method_names.get(fastest_method, fastest_method)} ({training_times[fastest_method]:.3f}s)")
            report_lines.append(f"  Most Memory Efficient: {self.method_names.get(most_efficient, most_efficient)} ({memory_usage[most_efficient]:.1f}MB)")
        
        # Save report
        report_path = f"{self.output_dir}/statistics_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Statistics report saved to {report_path}")
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("STATISTICS REPORT SUMMARY")
        print("=" * 80)
        for line in report_lines[:25]:  # Print first 25 lines
            print(line)
        if len(report_lines) > 25:
            print("... (see full report in statistics_report.txt)")
    
    def create_all_visualizations(self):
        """Create all visualizations."""
        print("🎨 Creating all visualizations...")
        
        # Create all plots
        self.plot_trajectory_comparison()
        self.plot_velocity_comparison()
        self.plot_acceleration_comparison()
        self.plot_performance_metrics()
        self.plot_method_comparison_summary()
        
        # Create all summaries and reports
        self.create_text_summary()
        self.create_csv_summary()
        self.create_statistics_report()
        
        print("\n✅ All visualizations completed!")
        print(f"📁 Output directory: {self.output_dir}")
        print("📊 Generated files:")
        print("  - trajectory_comparison.png")
        print("  - velocity_comparison.png")
        print("  - acceleration_comparison.png")
        print("  - performance_metrics.png")
        print("  - method_comparison_summary.png")
        print("  - text_summary.txt")
        print("  - trajectory_summary.csv")
        print("  - performance_summary.csv")
        print("  - statistics_report.txt")

def main():
    """Main function to run visualization."""
    print("🎨 Integrated Results Visualizer (FULL VERSION)")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = IntegratedResultsVisualizer()
    
    # Create all visualizations
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main()