"""
Evaluation Module
Compute comprehensive performance metrics
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_trajectory_tracking(actual_traj, desired_traj, dt=0.02):
    """
    Compute trajectory tracking metrics
    
    Args:
        actual_traj: Array of actual positions (N x 3)
        desired_traj: Array of desired positions (N x 3)
        dt: Sampling time (s)
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    actual = np.array(actual_traj)
    desired = np.array(desired_traj)
    
    # Position errors
    errors = actual - desired
    errors_norm = np.linalg.norm(errors, axis=1)
    
    # Component-wise errors
    x_errors = np.abs(errors[:, 0])
    y_errors = np.abs(errors[:, 1])
    z_errors = np.abs(errors[:, 2])
    
    # RMS errors
    rms_total = np.sqrt(np.mean(errors_norm**2))
    rms_x = np.sqrt(np.mean(x_errors**2))
    rms_y = np.sqrt(np.mean(y_errors**2))
    rms_z = np.sqrt(np.mean(z_errors**2))
    
    # Max errors
    max_total = np.max(errors_norm)
    max_x = np.max(x_errors)
    max_y = np.max(y_errors)
    max_z = np.max(z_errors)
    
    # Percentage errors (relative to trajectory range)
    x_range = np.max(desired[:, 0]) - np.min(desired[:, 0])
    y_range = np.max(desired[:, 1]) - np.min(desired[:, 1])
    z_range = np.max(desired[:, 2]) - np.min(desired[:, 2])
    total_range = np.sqrt(x_range**2 + y_range**2 + z_range**2)
    
    pct_x = (rms_x / x_range * 100) if x_range > 0 else 0
    pct_y = (rms_y / y_range * 100) if y_range > 0 else 0
    pct_z = (rms_z / z_range * 100) if z_range > 0 else 0
    pct_total = (rms_total / total_range * 100) if total_range > 0 else 0
    
    # Convergence time (time to get within 50mm and stay)
    threshold = 0.05  # 50mm
    below_threshold = errors_norm < threshold
    convergence_idx = 0
    for i in range(len(below_threshold) - 50):
        if np.all(below_threshold[i:i+50]):
            convergence_idx = i
            break
    convergence_time = convergence_idx * dt
    
    # Steady-state error (last 5 seconds)
    steady_state_samples = int(5.0 / dt)
    if len(errors_norm) >= steady_state_samples:
        steady_state_error = np.mean(errors_norm[-steady_state_samples:])
    else:
        steady_state_error = np.mean(errors_norm)
    
    # Overshoot
    max_error_first_half = np.max(errors_norm[:len(errors_norm)//2])
    
    metrics = {
        # RMS errors (mm)
        'rms_total_mm': rms_total * 1000,
        'rms_x_mm': rms_x * 1000,
        'rms_y_mm': rms_y * 1000,
        'rms_z_mm': rms_z * 1000,
        
        # Max errors (mm)
        'max_total_mm': max_total * 1000,
        'max_x_mm': max_x * 1000,
        'max_y_mm': max_y * 1000,
        'max_z_mm': max_z * 1000,
        
        # Percentage errors
        'pct_x': pct_x,
        'pct_y': pct_y,
        'pct_z': pct_z,
        'pct_total': pct_total,
        
        # Time-domain metrics
        'convergence_time_s': convergence_time,
        'steady_state_error_mm': steady_state_error * 1000,
        'max_overshoot_mm': max_error_first_half * 1000,
        
        # Trajectory characteristics
        'trajectory_length_m': total_range,
        'duration_s': len(actual) * dt
    }
    
    return metrics


def evaluate_attitude_performance(attitudes, attitudes_desired=None, dt=0.02):
    """
    Evaluate attitude control performance
    
    Args:
        attitudes: Array of attitudes in degrees (N x 3)
        attitudes_desired: Desired attitudes (N x 3), optional
        dt: Sampling time
        
    Returns:
        metrics: Dictionary of attitude metrics
    """
    att = np.array(attitudes)
    
    # Statistics
    mean_roll = np.mean(np.abs(att[:, 0]))
    mean_pitch = np.mean(np.abs(att[:, 1]))
    mean_yaw = np.mean(np.abs(att[:, 2]))
    
    max_roll = np.max(np.abs(att[:, 0]))
    max_pitch = np.max(np.abs(att[:, 1]))
    max_yaw = np.max(np.abs(att[:, 2]))
    
    std_roll = np.std(att[:, 0])
    std_pitch = np.std(att[:, 1])
    std_yaw = np.std(att[:, 2])
    
    metrics = {
        'mean_roll_deg': mean_roll,
        'mean_pitch_deg': mean_pitch,
        'mean_yaw_deg': mean_yaw,
        
        'max_roll_deg': max_roll,
        'max_pitch_deg': max_pitch,
        'max_yaw_deg': max_yaw,
        
        'std_roll_deg': std_roll,
        'std_pitch_deg': std_pitch,
        'std_yaw_deg': std_yaw,
    }
    
    # If desired attitudes provided, compute tracking errors
    if attitudes_desired is not None:
        att_des = np.array(attitudes_desired)
        att_errors = att - att_des
        
        # Wrap angles to [-180, 180]
        att_errors = np.mod(att_errors + 180, 360) - 180
        
        rms_roll = np.sqrt(np.mean(att_errors[:, 0]**2))
        rms_pitch = np.sqrt(np.mean(att_errors[:, 1]**2))
        rms_yaw = np.sqrt(np.mean(att_errors[:, 2]**2))
        
        metrics.update({
            'rms_roll_error_deg': rms_roll,
            'rms_pitch_error_deg': rms_pitch,
            'rms_yaw_error_deg': rms_yaw,
        })
    
    return metrics


def evaluate_control_effort(actions, dt=0.02):
    """
    Evaluate control effort and smoothness
    
    Args:
        actions: Array of actions/control commands (N x D)
        dt: Sampling time
        
    Returns:
        metrics: Dictionary of control metrics
    """
    actions = np.array(actions)
    
    # Control magnitude
    mean_action = np.mean(np.abs(actions), axis=0)
    max_action = np.max(np.abs(actions), axis=0)
    
    # Control smoothness (rate of change)
    action_rates = np.diff(actions, axis=0) / dt
    mean_rate = np.mean(np.abs(action_rates), axis=0)
    max_rate = np.max(np.abs(action_rates), axis=0)
    
    # Total variation (measure of smoothness)
    total_variation = np.sum(np.abs(action_rates), axis=0)
    
    metrics = {
        'mean_action_magnitude': mean_action,
        'max_action_magnitude': max_action,
        'mean_action_rate': mean_rate,
        'max_action_rate': max_rate,
        'total_variation': total_variation
    }
    
    return metrics


def print_evaluation_summary(traj_metrics, att_metrics=None, control_metrics=None):
    """
    Print formatted evaluation summary
    
    Args:
        traj_metrics: Trajectory tracking metrics
        att_metrics: Attitude performance metrics (optional)
        control_metrics: Control effort metrics (optional)
    """
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    # Trajectory tracking
    print("\n📍 TRAJECTORY TRACKING:")
    print(f"  RMS Error (3D): {traj_metrics['rms_total_mm']:.2f} mm ({traj_metrics['pct_total']:.3f}%)")
    print(f"    X-axis: {traj_metrics['rms_x_mm']:.2f} mm ({traj_metrics['pct_x']:.3f}%)")
    print(f"    Y-axis: {traj_metrics['rms_y_mm']:.2f} mm ({traj_metrics['pct_y']:.3f}%)")
    print(f"    Z-axis: {traj_metrics['rms_z_mm']:.2f} mm ({traj_metrics['pct_z']:.3f}%)")
    print(f"\n  Max Error: {traj_metrics['max_total_mm']:.2f} mm")
    print(f"    X: {traj_metrics['max_x_mm']:.2f} mm")
    print(f"    Y: {traj_metrics['max_y_mm']:.2f} mm")
    print(f"    Z: {traj_metrics['max_z_mm']:.2f} mm")
    print(f"\n  Convergence Time: {traj_metrics['convergence_time_s']:.2f} s")
    print(f"  Steady-State Error: {traj_metrics['steady_state_error_mm']:.2f} mm")
    print(f"  Max Overshoot: {traj_metrics['max_overshoot_mm']:.2f} mm")
    
    # Attitude
    if att_metrics:
        print("\n📐 ATTITUDE CONTROL:")
        print(f"  Mean Angles:")
        print(f"    Roll:  {att_metrics['mean_roll_deg']:.2f}° (max: {att_metrics['max_roll_deg']:.2f}°)")
        print(f"    Pitch: {att_metrics['mean_pitch_deg']:.2f}° (max: {att_metrics['max_pitch_deg']:.2f}°)")
        print(f"    Yaw:   {att_metrics['mean_yaw_deg']:.2f}° (max: {att_metrics['max_yaw_deg']:.2f}°)")
        
        if 'rms_roll_error_deg' in att_metrics:
            print(f"\n  RMS Tracking Errors:")
            print(f"    Roll:  {att_metrics['rms_roll_error_deg']:.2f}°")
            print(f"    Pitch: {att_metrics['rms_pitch_error_deg']:.2f}°")
            print(f"    Yaw:   {att_metrics['rms_yaw_error_deg']:.2f}°")
    
    # Control effort
    if control_metrics:
        print("\n🎮 CONTROL EFFORT:")
        print(f"  Mean Action Magnitude: {np.mean(control_metrics['mean_action_magnitude']):.3f}")
        print(f"  Max Action Magnitude: {np.max(control_metrics['max_action_magnitude']):.3f}")
        print(f"  Mean Action Rate: {np.mean(control_metrics['mean_action_rate']):.3f}")
        print(f"  Total Variation: {np.mean(control_metrics['total_variation']):.2f}")
    
    print("\n" + "="*70)
    
    # Compare with MATLAB baseline
    print("\n📊 COMPARISON WITH MATLAB BASELINE:")
    print("  MATLAB Results:")
    print("    X-axis: 0.91% error")
    print("    Y-axis: 1.05% error")
    print("    Z-axis: 0.11% error")
    print("    3D: 0.14% error")
    print("\n  Current Results:")
    print(f"    X-axis: {traj_metrics['pct_x']:.2f}% error {'✓' if traj_metrics['pct_x'] <= 1.0 else '✗'}")
    print(f"    Y-axis: {traj_metrics['pct_y']:.2f}% error {'✓' if traj_metrics['pct_y'] <= 1.1 else '✗'}")
    print(f"    Z-axis: {traj_metrics['pct_z']:.2f}% error {'✓' if traj_metrics['pct_z'] <= 0.2 else '✗'}")
    print(f"    3D: {traj_metrics['pct_total']:.2f}% error {'✓' if traj_metrics['pct_total'] <= 0.3 else '✗'}")
    print("="*70 + "\n")


def evaluate_full_episode(env, position_agent, attitude_agent=None):
    """
    Run a full evaluation episode and compute all metrics
    
    Args:
        env: PositionEnv instance
        position_agent: Trained position controller
        attitude_agent: Trained attitude controller (if separate)
        
    Returns:
        results: Dictionary with all evaluation data
    """
    print("\nRunning evaluation episode...")
    
    # Set to full trajectory, no exploration
    env.set_trajectory_scale(1.0)
    
    state = env.reset()
    done = False
    
    actual_trajectory = []
    desired_trajectory = []
    attitudes = []
    actions_position = []
    actions_attitude = []
    
    while not done:
        # Position control action (no exploration)
        pos_action = position_agent.select_action(state, explore=False)
        actions_position.append(pos_action)
        
        # Step environment
        next_state, reward, done, info = env.step(pos_action)
        
        # Record data
        actual_trajectory.append(env.state[0:3].copy())
        desired_trajectory.append(info.get('pos_des', np.zeros(3)))
        attitudes.append(np.rad2deg(env.state[3:6].copy()))
        
        state = next_state
    
    print("Episode complete. Computing metrics...")
    
    # Compute metrics
    traj_metrics = evaluate_trajectory_tracking(actual_trajectory, desired_trajectory)
    att_metrics = evaluate_attitude_performance(attitudes)
    control_metrics = evaluate_control_effort(actions_position)
    
    # Print summary
    print_evaluation_summary(traj_metrics, att_metrics, control_metrics)
    
    results = {
        'actual_trajectory': actual_trajectory,
        'desired_trajectory': desired_trajectory,
        'attitudes': attitudes,
        'actions_position': actions_position,
        'trajectory_metrics': traj_metrics,
        'attitude_metrics': att_metrics,
        'control_metrics': control_metrics
    }
    
    return results


def evaluate_system(env, position_agent, num_episodes=10, save_dir=None):
    """
    Evaluate the complete cascaded system over multiple episodes
    
    Args:
        env: PositionEnv instance
        position_agent: Trained position controller
        num_episodes: Number of evaluation episodes
        save_dir: Directory to save detailed results (optional)
        
    Returns:
        results: Dictionary with aggregated metrics
    """
    print(f"\nEvaluating system over {num_episodes} episodes...")
    
    all_pos_errors = []
    all_max_errors = []
    all_convergence_times = []
    all_max_rolls = []
    all_max_pitches = []
    all_mean_rolls = []
    all_mean_pitches = []
    success_count = 0
    
    for ep in range(num_episodes):
        print(f"  Episode {ep+1}/{num_episodes}...", end='')
        
        # Run episode
        state = env.reset()
        done = False
        
        ep_actual_traj = []
        ep_desired_traj = []
        ep_attitudes = []
        
        while not done:
            action = position_agent.select_action(state, explore=False)
            next_state, reward, done, info = env.step(action)
            
            ep_actual_traj.append(env.state[0:3].copy())
            ep_attitudes.append(np.rad2deg(env.state[3:6].copy()))
            
            state = next_state
        
        # Get desired trajectory
        ep_desired_traj = env.desired_trajectory_history
        
        # Compute metrics
        traj_metrics = evaluate_trajectory_tracking(ep_actual_traj, ep_desired_traj)
        att_metrics = evaluate_attitude_performance(ep_attitudes)
        
        # Aggregate
        all_pos_errors.append(traj_metrics['rms_total_mm'])
        all_max_errors.append(traj_metrics['max_total_mm'])
        all_convergence_times.append(traj_metrics['convergence_time_s'])
        all_max_rolls.append(att_metrics['max_roll_deg'])
        all_max_pitches.append(att_metrics['max_pitch_deg'])
        all_mean_rolls.append(att_metrics['mean_roll_deg'])
        all_mean_pitches.append(att_metrics['mean_pitch_deg'])
        
        # Success if < 30mm error
        if traj_metrics['rms_total_mm'] < 30:
            success_count += 1
        
        print(f" RMS: {traj_metrics['rms_total_mm']:.1f}mm")
    
    # Aggregate results
    results = {
        'mean_pos_error': np.mean(all_pos_errors),
        'std_pos_error': np.std(all_pos_errors),
        'rmse': np.sqrt(np.mean(np.array(all_pos_errors)**2)),
        'max_error': np.mean(all_max_errors),
        'mean_conv_time': np.mean(all_convergence_times),
        'success_rate': success_count / num_episodes,
        'max_roll': np.mean(all_max_rolls),
        'max_pitch': np.mean(all_max_pitches),
        'mean_roll': np.mean(all_mean_rolls),
        'mean_pitch': np.mean(all_mean_pitches),
        'all_errors': all_pos_errors,
    }
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Mean Position Error: {results['mean_pos_error']:.1f} ± {results['std_pos_error']:.1f} mm")
    print(f"RMSE: {results['rmse']:.1f} mm")
    print(f"Max Error: {results['max_error']:.1f} mm")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Mean Convergence Time: {results['mean_conv_time']:.2f} s")
    print(f"Max Roll: {results['max_roll']:.1f}°")
    print(f"Max Pitch: {results['max_pitch']:.1f}°")
    print(f"{'='*60}\n")
    
    return results


def save_evaluation_results(results, save_path):
    """
    Save evaluation results to file
    
    Args:
        results: Dictionary with evaluation data
        save_path: Path to save results
    """
    np.save(save_path, results)
    print(f"\nEvaluation results saved to {save_path}")