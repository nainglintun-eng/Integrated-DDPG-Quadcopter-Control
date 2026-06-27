"""
Visualization Module
Plotting functions for training results and trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_attitude_training(stats, save_path=None):
    """
    Plot attitude controller training curves
    
    Args:
        stats: Dictionary with training statistics
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Attitude Controller Training', fontsize=16, fontweight='bold')
    
    episodes = range(1, len(stats['rewards']) + 1)
    
    # Rewards
    axes[0, 0].plot(episodes, stats['rewards'], alpha=0.3, label='Episode')
    axes[0, 0].plot(episodes, moving_average(stats['rewards'], 100), 
                   linewidth=2, label='Moving Avg (100)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Attitude Error
    axes[0, 1].plot(episodes, stats['att_errors'], alpha=0.3, label='Episode')
    axes[0, 1].plot(episodes, moving_average(stats['att_errors'], 100),
                   linewidth=2, label='Moving Avg (100)')
    axes[0, 1].axhline(y=2.0, color='r', linestyle='--', label='Target (2°)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Attitude Error (degrees)')
    axes[0, 1].set_title('Attitude Tracking Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rate Error
    axes[1, 0].plot(episodes, stats['rate_errors'], alpha=0.3, label='Episode')
    axes[1, 0].plot(episodes, moving_average(stats['rate_errors'], 100),
                   linewidth=2, label='Moving Avg (100)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Rate Error (rad/s)')
    axes[1, 0].set_title('Angular Rate Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Success Rate
    success_rate = [np.mean(stats['successes'][max(0, i-99):i+1]) 
                   for i in range(len(stats['successes']))]
    axes[1, 1].plot(episodes, success_rate, linewidth=2)
    axes[1, 1].axhline(y=0.95, color='g', linestyle='--', label='Target (95%)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title('Success Rate (100-episode window)')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attitude training plot to {save_path}")
        plt.close(fig)  # ADD THIS LIN
    
    return fig


def plot_position_training(stats, save_path=None):
    """
    Plot position controller training curves with curriculum phases
    
    Args:
        stats: Dictionary with training statistics including 'phases'
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Position Controller Training (Curriculum Learning)', 
                fontsize=16, fontweight='bold')
    
    episodes = range(1, len(stats['rewards']) + 1)
    phases = stats.get('phases', [1] * len(stats['rewards']))
    
    # Color by phase
    phase_colors = ['blue', 'green', 'orange', 'red']
    
    # Rewards
    for phase in range(1, max(phases) + 1):
        phase_mask = [p == phase for p in phases]
        phase_episodes = [e for e, m in zip(episodes, phase_mask) if m]
        phase_rewards = [r for r, m in zip(stats['rewards'], phase_mask) if m]
        axes[0, 0].plot(phase_episodes, phase_rewards, alpha=0.3, 
                       color=phase_colors[phase-1])
    
    axes[0, 0].plot(episodes, moving_average(stats['rewards'], 100),
                   linewidth=2, color='black', label='Moving Avg (100)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add phase boundaries
    phase_changes = [i for i in range(1, len(phases)) if phases[i] != phases[i-1]]
    for pc in phase_changes:
        axes[0, 0].axvline(x=pc, color='gray', linestyle='--', alpha=0.5)
    
    # Position Error
    for phase in range(1, max(phases) + 1):
        phase_mask = [p == phase for p in phases]
        phase_episodes = [e for e, m in zip(episodes, phase_mask) if m]
        phase_errors = [r for r, m in zip(stats['pos_errors'], phase_mask) if m]
        axes[0, 1].plot(phase_episodes, phase_errors, alpha=0.3,
                       color=phase_colors[phase-1], label=f'Phase {phase}')
    
    axes[0, 1].plot(episodes, moving_average(stats['pos_errors'], 100),
                   linewidth=2, color='black', label='Moving Avg')
    axes[0, 1].axhline(y=20, color='r', linestyle='--', label='Target (20mm)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Position Error (mm)')
    axes[0, 1].set_title('Position Tracking Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    for pc in phase_changes:
        axes[0, 1].axvline(x=pc, color='gray', linestyle='--', alpha=0.5)
    
    # Velocity Error
    axes[1, 0].plot(episodes, stats['vel_errors'], alpha=0.3, label='Episode')
    axes[1, 0].plot(episodes, moving_average(stats['vel_errors'], 100),
                   linewidth=2, label='Moving Avg (100)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Velocity Error (m/s)')
    axes[1, 0].set_title('Velocity Tracking Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    for pc in phase_changes:
        axes[1, 0].axvline(x=pc, color='gray', linestyle='--', alpha=0.5)
    
    # Success Rate
    success_rate = [np.mean(stats['successes'][max(0, i-99):i+1])
                   for i in range(len(stats['successes']))]
    axes[1, 1].plot(episodes, success_rate, linewidth=2)
    axes[1, 1].axhline(y=0.90, color='g', linestyle='--', label='Target (90%)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title('Success Rate (100-episode window)')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    for pc in phase_changes:
        axes[1, 1].axvline(x=pc, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved position training plot to {save_path}")
        plt.close(fig)
    
    return fig


def plot_finetuning(stats, save_path=None):
    """
    Plot fine-tuning curves
    
    Args:
        stats: Dictionary with fine-tuning statistics
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Joint Fine-Tuning', fontsize=16, fontweight='bold')
    
    episodes = range(1, len(stats['rewards']) + 1)
    
    # Rewards
    axes[0, 0].plot(episodes, stats['rewards'], alpha=0.3, label='Episode')
    axes[0, 0].plot(episodes, moving_average(stats['rewards'], 50),
                   linewidth=2, label='Moving Avg (50)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Fine-Tuning Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Position and Attitude Errors
    axes[0, 1].plot(episodes, stats['pos_errors'], alpha=0.5, label='Position Error (mm)')
    axes[0, 1].plot(episodes, moving_average(stats['pos_errors'], 50),
                   linewidth=2, label='Position MA')
    ax2 = axes[0, 1].twinx()
    ax2.plot(episodes, stats['att_errors'], alpha=0.5, color='orange', 
            label='Attitude Error (°)')
    ax2.plot(episodes, moving_average(stats['att_errors'], 50),
            linewidth=2, color='red', label='Attitude MA')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Position Error (mm)', color='blue')
    ax2.set_ylabel('Attitude Error (°)', color='red')
    axes[0, 1].set_title('Position & Attitude Errors')
    axes[0, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Losses
    if stats['position_losses'] and stats['attitude_losses']:
        axes[1, 0].plot(episodes, moving_average(stats['position_losses'], 20),
                       linewidth=2, label='Position Actor Loss')
        axes[1, 0].plot(episodes, moving_average(stats['attitude_losses'], 20),
                       linewidth=2, label='Attitude Actor Loss')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Actor Loss')
        axes[1, 0].set_title('Actor Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Success Rate
    success_rate = [np.mean(stats['successes'][max(0, i-49):i+1])
                   for i in range(len(stats['successes']))]
    axes[1, 1].plot(episodes, success_rate, linewidth=2)
    axes[1, 1].axhline(y=0.90, color='g', linestyle='--', label='Target (90%)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title('Success Rate (50-episode window)')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved fine-tuning plot to {save_path}")
        plt.close(fig)
    
    return fig


def plot_3d_trajectory(actual_traj, desired_traj, attitudes=None, save_path=None):
    """
    Plot 3D trajectory comparison
    
    Args:
        actual_traj: Array of actual positions (N x 3)
        desired_traj: Array of desired positions (N x 3)
        attitudes: Array of attitudes in degrees (N x 3), optional
        save_path: Path to save figure (optional)
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 3D trajectory
    ax1 = fig.add_subplot(221, projection='3d')
    
    actual_traj = np.array(actual_traj)
    desired_traj = np.array(desired_traj)
    
    ax1.plot(desired_traj[:, 0], desired_traj[:, 1], desired_traj[:, 2],
            'b--', linewidth=2, label='Desired', alpha=0.7)
    ax1.plot(actual_traj[:, 0], actual_traj[:, 1], actual_traj[:, 2],
            'r-', linewidth=1.5, label='Actual')
    
    # Mark start and end
    ax1.scatter([desired_traj[0, 0]], [desired_traj[0, 1]], [desired_traj[0, 2]],
               c='green', s=100, marker='o', label='Start')
    ax1.scatter([desired_traj[-1, 0]], [desired_traj[-1, 1]], [desired_traj[-1, 2]],
               c='black', s=100, marker='x', label='End')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # XY projection
    ax2 = fig.add_subplot(222)
    ax2.plot(desired_traj[:, 0], desired_traj[:, 1], 'b--', linewidth=2, label='Desired')
    ax2.plot(actual_traj[:, 0], actual_traj[:, 1], 'r-', linewidth=1.5, label='Actual')
    ax2.scatter([desired_traj[0, 0]], [desired_traj[0, 1]], c='green', s=100, marker='o')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane (Top View)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Position errors over time
    ax3 = fig.add_subplot(223)
    errors = np.linalg.norm(actual_traj - desired_traj, axis=1) * 1000  # mm
    time = np.arange(len(errors)) * 0.02  # 50Hz
    ax3.plot(time, errors, linewidth=1.5)
    ax3.axhline(y=20, color='r', linestyle='--', label='Target (20mm)')
    ax3.axhline(y=np.mean(errors), color='g', linestyle='--', 
               label=f'Mean ({np.mean(errors):.1f}mm)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error (mm)')
    ax3.set_title('Position Tracking Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Attitudes if provided
    if attitudes is not None:
        ax4 = fig.add_subplot(224)
        attitudes = np.array(attitudes)
        time = np.arange(len(attitudes)) * 0.02
        ax4.plot(time, attitudes[:, 0], label='Roll (φ)', linewidth=1.5)
        ax4.plot(time, attitudes[:, 1], label='Pitch (θ)', linewidth=1.5)
        ax4.plot(time, attitudes[:, 2], label='Yaw (ψ)', linewidth=1.5)
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Angle (degrees)')
        ax4.set_title('Attitude Angles')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D trajectory plot to {save_path}")
        #plt.close(fig)
    
    return fig


def plot_training_curves(stats, save_path, title="Training Progress"):
    """
    Generic training curves plotter
    
    Args:
        stats: Dictionary with training statistics
        save_path: Path to save figure
        title: Plot title
    """
    # Determine which type of training based on keys
    if 'att_errors' in stats:
        return plot_attitude_training(stats, save_path)
    elif 'phases' in stats:
        return plot_position_training(stats, save_path)
    elif 'position_losses' in stats and 'attitude_losses' in stats:
        return plot_finetuning(stats, save_path)
    else:
        # Generic plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        episodes = range(1, len(stats['rewards']) + 1)
        
        # Rewards
        axes[0, 0].plot(episodes, stats['rewards'], alpha=0.3)
        axes[0, 0].plot(episodes, moving_average(stats['rewards'], 100), linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Position errors if available
        if 'pos_errors' in stats:
            axes[0, 1].plot(episodes, stats['pos_errors'], alpha=0.3)
            axes[0, 1].plot(episodes, moving_average(stats['pos_errors'], 100), linewidth=2)
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Position Error (mm)')
            axes[0, 1].set_title('Position Error')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Success rate
        if 'successes' in stats:
            success_rate = [np.mean(stats['successes'][max(0, i-99):i+1])
                           for i in range(len(stats['successes']))]
            axes[1, 0].plot(episodes, success_rate, linewidth=2)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].set_title('Success Rate (100-ep window)')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        if 'episode_lengths' in stats:
            axes[1, 1].plot(episodes, stats['episode_lengths'], alpha=0.5)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Steps')
            axes[1, 1].set_title('Episode Length')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training plot to {save_path}")
        plt.close()
        
        return fig


def plot_trajectory_3d(env, save_path):
    """
    Plot 3D trajectory from environment history
    
    Args:
        env: Environment with trajectory_history
        save_path: Path to save figure
    """
    if not hasattr(env, 'trajectory_history') or len(env.trajectory_history) == 0:
        print("Warning: No trajectory history in environment")
        return None
    
    actual = np.array(env.trajectory_history)
    desired = np.array(env.desired_trajectory_history) if hasattr(env, 'desired_trajectory_history') else None
    attitudes = np.array(env.attitude_history) if hasattr(env, 'attitude_history') else None
    
    return plot_3d_trajectory(actual, desired, attitudes, save_path)


def moving_average(data, window):
    """Compute moving average with same length as input"""
    if len(data) < window:
        return data
    
    # Pad the beginning to maintain same length
    cumsum = np.cumsum(np.insert(data, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    
    # Pad beginning with first value repeated
    padding = np.full(window-1, ma[0] if len(ma) > 0 else 0)
    result = np.concatenate([padding, ma])
    
    return result


def save_all_plots(attitude_stats, position_stats, finetune_stats, 
                   trajectory_data, save_dir):
    """
    Save all training plots
    
    Args:
        attitude_stats: Attitude training statistics
        position_stats: Position training statistics
        finetune_stats: Fine-tuning statistics
        trajectory_data: Dict with 'actual', 'desired', 'attitudes'
        save_dir: Directory to save plots
    """
    os.makedirs(f'{save_dir}/plots', exist_ok=True)
    
    print("\nGenerating training plots...")
    
    # Attitude training
    plot_attitude_training(attitude_stats, 
                          f'{save_dir}/plots/attitude_training.png')
    
    # Position training
    plot_position_training(position_stats,
                          f'{save_dir}/plots/position_training.png')
    
    # Fine-tuning
    plot_finetuning(finetune_stats,
                   f'{save_dir}/plots/finetuning.png')
    
    # 3D trajectory
    plot_3d_trajectory(trajectory_data['actual'],
                      trajectory_data['desired'],
                      trajectory_data.get('attitudes'),
                      f'{save_dir}/plots/final_trajectory.png')
    
    print(f"All plots saved to {save_dir}/plots/")