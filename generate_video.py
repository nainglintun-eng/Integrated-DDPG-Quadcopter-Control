"""
generate_integrated_video.py

Generates a video visualization of the trained integrated DDPG quadcopter
following the expanding helix trajectory.

This is a SINGLE-AGENT controller that directly outputs:
    U = [Ft, tau_x, tau_y, tau_z]
from a single DDPG agent with 18-dim observation space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import art3d
import torch
import sys
import os
from datetime import datetime
import warnings

# Suppress Gym deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import SystemConfig, IntegratedAgentConfig, get_trajectory_function
from agents.ddpg_agent import DDPGAgent
from environments.integrated_env import IntegratedEnv


class IntegratedDDPGVisualizer:
    """
    Generates video visualization of integrated DDPG quadcopter trajectory simulation.
    
    Key differences from cascaded version:
        - Single agent (no separate attitude controller)
        - Direct output: [Ft, tau_x, tau_y, tau_z]
        - Observation includes position, velocity, attitude, rates, desired trajectory
    """
    
    def __init__(self, agent_path, output_dir='integrated_videos'):
        self.sys_cfg = SystemConfig()
        self.int_cfg = IntegratedAgentConfig()
        
        # Load agent
        print("Loading trained integrated DDPG agent...")
        self.agent = self._load_agent(agent_path)
            
        # Create environment
        self.env = IntegratedEnv(config=self.int_cfg)
        self.get_trajectory = get_trajectory_function()
        
        # Setup output
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Simulation storage
        self.reset_simulation()
        
    def _load_agent(self, path):
        """Load integrated DDPG agent"""
        agent = DDPGAgent(
            state_dim=self.int_cfg.STATE_DIM,
            action_dim=self.int_cfg.ACTION_DIM,
            max_action=self.int_cfg.ACTION_SCALE,
            hidden_dims=self.int_cfg.HIDDEN_DIMS
        )
        
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location='cpu')
                agent.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
                agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'], strict=False)
                agent.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)
                agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'], strict=False)
                print(f"  ✓ Loaded integrated DDPG agent from: {path}")
            except Exception as e:
                print(f"  ⚠ Could not load agent: {e}")
        else:
            print(f"  ⚠ Agent not found: {path}")
            
        return agent
    
    def reset_simulation(self):
        """Reset simulation state"""
        self.positions = []
        self.desired_positions = []
        self.attitudes = []
        self.velocities = []
        self.actions = []
        self.thrusts = []
        self.torques = []
        self.times = []
        self.errors = []
        
        # Additional telemetry
        self.angular_rates = []
        self.action_norm = []
        
    # ------------------------------------------------------------------
    # Animation save helper
    # ------------------------------------------------------------------
    @staticmethod
    def _save_animation(anim, output_path, fps, dpi):
        """
        Save animation preferring MP4 (ffmpeg) over GIF (pillow).
        MP4 streams frames to disk — no RAM blow-up at 60 fps.
        Falls back to GIF at reduced DPI if ffmpeg is unavailable.
        """
        import shutil
        from matplotlib.animation import FFMpegWriter, PillowWriter

        base, _ = os.path.splitext(output_path)

        if shutil.which('ffmpeg'):
            mp4_path = base + '.mp4'
            writer = FFMpegWriter(
                fps=fps,
                metadata={'title': os.path.basename(base)},
                extra_args=[
                    '-vcodec', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '18',
                    '-preset', 'fast',
                ]
            )
            anim.save(mp4_path, writer=writer, dpi=dpi)
            print(f"  Saved MP4: {mp4_path}")
            return mp4_path

        print("  ffmpeg not found — falling back to GIF (reduced DPI to save RAM)")
        gif_dpi = min(dpi, 80)
        gif_path = base + '.gif'
        writer = PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=gif_dpi)
        print(f"  Saved GIF (dpi={gif_dpi}): {gif_path}")
        return gif_path

    # ------------------------------------------------------------------
    # Drone shape helper
    # ------------------------------------------------------------------
    @staticmethod
    def _drone_geometry(pos, att_deg, arm_len=0.22, rotor_r=0.08):
        """
        Return the geometry of an X-frame quadcopter at *pos* with
        Euler angles *att_deg* (degrees: roll, pitch, yaw).

        X-frame layout (top view):
              M1(+x+y)  M2(-x+y)
                  \\    /
                   [  ]   <- central body box
                  /    \\
              M4(+x-y)  M3(-x-y)

        Returns
        -------
        arms        : list of 4 (xs,ys,zs) – arm line segments
        rotors      : list of 4 (xs,ys,zs) – rotor disc circles
        body_lines  : list of 2 (xs,ys,zs) – body box outline + heading indicator
        motor_dots  : list of 4 (x,y,z)    – motor hub centres (for scatter)
        """
        roll, pitch, yaw = np.radians(att_deg)

        cy, sy = np.cos(yaw),   np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll),  np.sin(roll)
        R = np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,    cp*sr,              cp*cr            ]
        ])

        # X-frame: arm tips at 45° diagonals
        s = arm_len / np.sqrt(2)
        tips_body = np.array([
            [ s,  s, 0],   # front-right  M1
            [-s,  s, 0],   # front-left   M2
            [-s, -s, 0],   # rear-left    M3
            [ s, -s, 0],   # rear-right   M4
        ])
        tips_world = (R @ tips_body.T).T + pos

        # Arms (centre -> each motor hub)
        arms = []
        for tip in tips_world:
            arms.append(([pos[0], tip[0]],
                         [pos[1], tip[1]],
                         [pos[2], tip[2]]))

        # Rotor discs (flat circles in drone's XY plane)
        theta = np.linspace(0, 2 * np.pi, 24)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotors = []
        for tip in tips_world:
            circle_body = np.stack([rotor_r * cos_t,
                                    rotor_r * sin_t,
                                    np.zeros_like(theta)], axis=1)
            circle_world = (R @ circle_body.T).T + tip
            rotors.append((circle_world[:, 0],
                           circle_world[:, 1],
                           circle_world[:, 2]))

        # Central body – rectangular box outline
        bw, bl = 0.07, 0.09
        corners_body = np.array([
            [ bl,  bw, 0],
            [-bl,  bw, 0],
            [-bl, -bw, 0],
            [ bl, -bw, 0],
            [ bl,  bw, 0],   # close the loop
        ])
        corners_world = (R @ corners_body.T).T + pos
        body_lines = [(corners_world[:, 0],
                       corners_world[:, 1],
                       corners_world[:, 2])]

        # Forward heading indicator
        fwd_body = np.array([[0, 0, 0], [bl * 1.8, 0, 0]])
        fwd_world = (R @ fwd_body.T).T + pos
        body_lines.append(([fwd_world[0, 0], fwd_world[1, 0]],
                           [fwd_world[0, 1], fwd_world[1, 1]],
                           [fwd_world[0, 2], fwd_world[1, 2]]))

        # Motor hub dots
        motor_dots = [(tip[0], tip[1], tip[2]) for tip in tips_world]

        return arms, rotors, body_lines, motor_dots

    def run_simulation(self, duration=50.0):
        """
        Run full simulation
        
        Args:
            duration: Simulation duration in seconds
        """
        print(f"\nRunning {duration}s simulation with integrated DDPG agent...")
        
        # Reset environment
        obs = self.env.reset()
        self.env.set_trajectory_scale(1.0)
        self.env.set_wind([0.0, 0.0, 0.0])  # No wind for video
        
        # Initialize tracking
        time = 0.0
        step_count = 0
        max_steps = int(duration / self.sys_cfg.DT)
        
        while time < duration and step_count < max_steps:
            # Get action from integrated agent
            action = self.agent.select_action(obs, explore=False)
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            
            # Store data
            pos = self.env.state[0:3].copy()
            att = self.env.state[6:9].copy()
            vel = self.env.state[3:6].copy()
            
            pos_des, vel_des, _ = self.get_trajectory(time, 1.0)
            error = np.linalg.norm(pos - pos_des)
            
            # Extract physical control outputs
            Ft = info.get('Ft', 0)
            torques = info.get('torques', np.zeros(3))
            
            self.positions.append(pos)
            self.desired_positions.append(pos_des)
            self.attitudes.append(np.degrees(att))  # Convert to degrees
            self.velocities.append(vel)
            self.actions.append(action.copy())
            self.thrusts.append(Ft)
            self.torques.append(torques)
            self.times.append(time)
            self.errors.append(error)
            self.angular_rates.append(np.degrees(self.env.state[9:12].copy()))
            self.action_norm.append(np.linalg.norm(action))
            
            # Progress indicator
            if step_count % 1000 == 0:
                print(f"  Progress: {time:.1f}s, Error: {error*1000:.1f}mm, Ft: {Ft:.2f}N")
            
            if done:
                print(f"  Episode terminated at t={time:.2f}s: {info.get('termination', 'unknown')}")
                break
                
            time += self.sys_cfg.DT
            step_count += 1
            
        print(f"  Simulation complete: {len(self.times)} steps, final error={self.errors[-1]*1000:.1f}mm")
        
        # Convert to numpy arrays for easier manipulation
        self.positions = np.array(self.positions)
        self.desired_positions = np.array(self.desired_positions)
        self.attitudes = np.array(self.attitudes)
        self.velocities = np.array(self.velocities)
        self.actions = np.array(self.actions)
        self.torques = np.array(self.torques)
        self.times = np.array(self.times)
        self.errors = np.array(self.errors)
        self.angular_rates = np.array(self.angular_rates)
        
        return self
    
    def create_3d_animation(self, filename='integrated_trajectory.mp4', 
                           fps=60, interval=17, dpi=150):
        """
        Create 3D trajectory animation with fixed camera (no rotation)
        """
        print(f"\nCreating 3D animation: {filename}")
        
        # Subsampling for performance (show every N steps)
        duration = self.times[-1]
        total_frames = int(duration * fps)
        step = max(1, len(self.times) // total_frames)
        
        times_sub = self.times[::step]
        pos_sub = self.positions[::step]
        des_sub = self.desired_positions[::step]
        
        # Setup figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot full trajectories in background
        ax.plot3D(self.positions[:,0], self.positions[:,1], self.positions[:,2], 
                 'b-', alpha=0.3, linewidth=1, label='Actual Trajectory')
        ax.plot3D(self.desired_positions[:,0], self.desired_positions[:,1], 
                 self.desired_positions[:,2], 'r--', alpha=0.3, linewidth=1, 
                 label='Desired Trajectory')
        
        # Starting point markers
        ax.scatter(*self.positions[0,:3], color='green', s=100, marker='o', 
                  label='Start')
        ax.scatter(*self.desired_positions[0,:3], color='orange', s=100, 
                  marker='^', label='Desired Start')
        
        # Quadcopter representation – X-frame drone shape
        drone_arm_lines   = [ax.plot3D([], [], [], color='#333333', linewidth=2.5)[0] for _ in range(4)]
        drone_rotor_lines = [ax.plot3D([], [], [], color='#1a88cc', linewidth=1.2, alpha=0.85)[0] for _ in range(4)]
        drone_body_lines  = [ax.plot3D([], [], [], color='#222222', linewidth=2)[0] for _ in range(2)]
        drone_motors      = ax.scatter([], [], [], color='red', s=18, zorder=5)
        quad, = ax.plot3D([], [], [], 'bo', markersize=4, label='Quadcopter')  # legend proxy
        
        # Trajectory trail
        trail_length = min(50, len(pos_sub))
        trail_line, = ax.plot3D([], [], [], 'b-', alpha=0.5, linewidth=2)
        
        # Desired position marker
        des_marker, = ax.plot3D([], [], [], 'ro', markersize=10, alpha=0.8)
        
        # Error line
        error_line, = ax.plot3D([], [], [], 'g--', linewidth=1, alpha=0.6)
        
        # Set axis labels and limits
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('Integrated DDPG Quadcopter Trajectory Tracking\nExpanding Helix Mission', 
                    fontsize=14, fontweight='bold')
        
        # Set limits (fixed, don't change during animation)
        max_xy = max(np.max(np.abs(self.positions[:,0])), 
                    np.max(np.abs(self.positions[:,1])),
                    np.max(self.desired_positions[:,0]),
                    np.max(self.desired_positions[:,1])) + 1
        ax.set_xlim(-max_xy, max_xy)
        ax.set_ylim(-max_xy, max_xy)
        z_min = max(0, np.min(self.positions[:,2]) - 2)
        z_max = np.max(self.positions[:,2]) + 2
        ax.set_zlim(z_min, z_max)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        # Text display for metrics
        text_box = ax.text2D(0.02, 0.98, '', transform=ax.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Fixed camera angle (no rotation)
        ax.view_init(elev=30, azim=45)
        
        # Animation function
        def animate(frame):
            idx = frame
            if idx >= len(pos_sub):
                idx = len(pos_sub) - 1
                
            # Update quadcopter – X-frame drone shape
            att_idx = min(idx, len(self.attitudes) - 1)
            arms, rotors, body_ls, motor_dots = self._drone_geometry(
                pos_sub[idx], self.attitudes[att_idx])
            for i, (xs, ys, zs) in enumerate(arms):
                drone_arm_lines[i].set_data_3d(xs, ys, zs)
            for i, (xs, ys, zs) in enumerate(rotors):
                drone_rotor_lines[i].set_data_3d(xs, ys, zs)
            for i, (xs, ys, zs) in enumerate(body_ls):
                drone_body_lines[i].set_data_3d(xs, ys, zs)
            drone_motors._offsets3d = ([d[0] for d in motor_dots],
                                       [d[1] for d in motor_dots],
                                       [d[2] for d in motor_dots])
            quad.set_data_3d([pos_sub[idx, 0]], [pos_sub[idx, 1]], [pos_sub[idx, 2]])
            
            # Update desired marker
            des_marker.set_data_3d([des_sub[idx,0]], [des_sub[idx,1]], [des_sub[idx,2]])
            
            # Update trail (last 50 positions)
            trail_start = max(0, idx - trail_length)
            trail_line.set_data_3d(pos_sub[trail_start:idx+1,0],
                                  pos_sub[trail_start:idx+1,1],
                                  pos_sub[trail_start:idx+1,2])
            
            # Update error line
            error_line.set_data_3d([pos_sub[idx,0], des_sub[idx,0]],
                                  [pos_sub[idx,1], des_sub[idx,1]],
                                  [pos_sub[idx,2], des_sub[idx,2]])
            
            # Update text
            t = times_sub[idx]
            err = np.linalg.norm(pos_sub[idx] - des_sub[idx])
            text_box.set_text(f'Integrated DDPG Controller\n'
                            f'Time: {t:.1f}s\n'
                            f'Position Error: {err*1000:.1f}mm\n'
                            f'Altitude: {pos_sub[idx,2]:.2f}m\n'
                            f'Progress: {t/duration*100:.0f}%')
            
            return (*drone_arm_lines, *drone_rotor_lines, *drone_body_lines,
                    drone_motors, quad, des_marker, trail_line, error_line, text_box)
        
        # Create animation
        frames = min(len(pos_sub), total_frames)
        anim = FuncAnimation(fig, animate, frames=frames, interval=interval, 
                            blit=False, repeat=False)
        
        # Save animation (MP4 via ffmpeg if available, else GIF fallback)
        output_path = os.path.join(self.output_dir, filename)
        self._save_animation(anim, output_path, fps, dpi)
        
        plt.close(fig)
        return anim
    
    def create_dashboard_animation(self, filename='integrated_dashboard.mp4',
                                  fps=60, interval=17, dpi=120):
        """
        Create comprehensive dashboard animation with fixed camera
        """
        print(f"\nCreating dashboard animation: {filename}")
        
        # Subsampling
        duration = self.times[-1]
        total_frames = int(duration * fps)
        step = max(1, len(self.times) // total_frames)
        
        # Setup figure
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 3D Trajectory (spanning top-left)
        ax_3d = fig.add_subplot(gs[:2, :2], projection='3d')
        
        # Attitude plot
        ax_att = fig.add_subplot(gs[0, 2])
        
        # Position error plot
        ax_err = fig.add_subplot(gs[1, 2])
        
        # Thrust plot
        ax_thrust = fig.add_subplot(gs[2, 0])
        
        # Torques plot
        ax_torques = fig.add_subplot(gs[2, 1])
        
        # Telemetry text
        ax_text = fig.add_subplot(gs[2, 2])
        ax_text.axis('off')
        
        # Initialize 3D plot elements
        # Full trajectories
        ax_3d.plot3D(self.positions[:,0], self.positions[:,1], self.positions[:,2], 
                    'b-', alpha=0.3, linewidth=1)
        ax_3d.plot3D(self.desired_positions[:,0], self.desired_positions[:,1], 
                    self.desired_positions[:,2], 'r--', alpha=0.3, linewidth=1)
        
        # Dynamic elements – X-frame drone shape
        drone_arm_lines3d   = [ax_3d.plot3D([], [], [], color='#333333', linewidth=2.5)[0] for _ in range(4)]
        drone_rotor_lines3d = [ax_3d.plot3D([], [], [], color='#1a88cc', linewidth=1.2, alpha=0.85)[0] for _ in range(4)]
        drone_body_lines3d  = [ax_3d.plot3D([], [], [], color='#222222', linewidth=2)[0] for _ in range(2)]
        drone_motors3d      = ax_3d.scatter([], [], [], color='red', s=18, zorder=5)
        quad_3d, = ax_3d.plot3D([], [], [], 'bo', markersize=4)  # legend proxy
        des_3d, = ax_3d.plot3D([], [], [], 'ro', markersize=10)
        trail_line, = ax_3d.plot3D([], [], [], 'b-', alpha=0.5, linewidth=2)
        
        # 3D plot labels
        ax_3d.set_xlabel('X (m)', fontsize=10)
        ax_3d.set_ylabel('Y (m)', fontsize=10)
        ax_3d.set_zlabel('Z (m)', fontsize=10)
        ax_3d.set_title('Integrated DDPG Trajectory Tracking', fontweight='bold', fontsize=11)
        
        # Set 3D limits (fixed, don't change)
        max_xy = max(np.max(np.abs(self.positions[:,0])), 
                    np.max(np.abs(self.positions[:,1]))) + 1
        ax_3d.set_xlim(-max_xy, max_xy)
        ax_3d.set_ylim(-max_xy, max_xy)
        ax_3d.set_zlim(0, np.max(self.positions[:,2]) + 2)
        
        # Fixed camera angle (no rotation)
        ax_3d.view_init(elev=25, azim=45)
        
        # Attitude plot setup
        ax_att.set_xlim(0, duration)
        ax_att.set_ylim(-60, 60)
        ax_att.set_xlabel('Time (s)', fontsize=9)
        ax_att.set_ylabel('Angle (deg)', fontsize=9)
        ax_att.set_title('Integrated DDPG Attitude', fontweight='bold', fontsize=10)
        ax_att.grid(True, alpha=0.3)
        roll_line, = ax_att.plot([], [], 'r-', label='Roll', linewidth=1.5)
        pitch_line, = ax_att.plot([], [], 'g-', label='Pitch', linewidth=1.5)
        yaw_line, = ax_att.plot([], [], 'b-', label='Yaw', linewidth=1.5)
        ax_att.legend(loc='upper right', fontsize=7)
        
        # Angular rates plot (secondary on same axis)
        ax_att2 = ax_att.twinx()
        ax_att2.set_ylabel('Angular Rate (deg/s)', fontsize=8, color='purple')
        ax_att2.tick_params(axis='y', labelcolor='purple')
        rate_p_line, = ax_att2.plot([], [], 'r--', linewidth=1, alpha=0.5, label='p rate')
        rate_q_line, = ax_att2.plot([], [], 'g--', linewidth=1, alpha=0.5, label='q rate')
        
        # Position error plot
        ax_err.set_xlim(0, duration)
        max_error_mm = max(self.errors * 1000) + 100
        ax_err.set_ylim(0, max_error_mm)
        ax_err.set_xlabel('Time (s)', fontsize=9)
        ax_err.set_ylabel('Error (mm)', fontsize=9)
        ax_err.set_title('Integrated DDPG Tracking Error', fontweight='bold', fontsize=10)
        ax_err.grid(True, alpha=0.3)
        err_line, = ax_err.plot([], [], 'b-', linewidth=1.5)
        # Add threshold lines
        ax_err.axhline(y=500, color='y', linestyle='--', alpha=0.5, linewidth=0.8)
        ax_err.axhline(y=200, color='g', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Thrust plot
        ax_thrust.set_xlim(0, duration)
        thrust_min = self.sys_cfg.MIN_THRUST
        thrust_max = self.sys_cfg.MAX_THRUST
        ax_thrust.set_ylim(thrust_min - 1, thrust_max + 1)
        ax_thrust.set_xlabel('Time (s)', fontsize=9)
        ax_thrust.set_ylabel('Thrust (N)', fontsize=9)
        ax_thrust.set_title('Integrated DDPG Total Thrust', fontweight='bold', fontsize=10)
        ax_thrust.grid(True, alpha=0.3)
        thrust_line, = ax_thrust.plot([], [], 'm-', linewidth=1.5)
        ax_thrust.axhline(y=self.sys_cfg.MASS * self.sys_cfg.GRAVITY, 
                         color='k', linestyle='--', alpha=0.5, linewidth=0.8, label='Hover')
        
        # Torques plot
        ax_torques.set_xlim(0, duration)
        max_torque = self.sys_cfg.MAX_TORQUE
        ax_torques.set_ylim(-max_torque, max_torque)
        ax_torques.set_xlabel('Time (s)', fontsize=9)
        ax_torques.set_ylabel('Torque (Nm)', fontsize=9)
        ax_torques.set_title('Integrated DDPG Control Torques', fontweight='bold', fontsize=10)
        ax_torques.grid(True, alpha=0.3)
        torque_x_line, = ax_torques.plot([], [], 'r-', label='τ_x', linewidth=1)
        torque_y_line, = ax_torques.plot([], [], 'g-', label='τ_y', linewidth=1)
        torque_z_line, = ax_torques.plot([], [], 'b-', label='τ_z', linewidth=1)
        ax_torques.legend(loc='upper right', fontsize=7)
        
        # Animation function
        def animate(frame):
            # Get current index
            idx = frame * step
            if idx >= len(self.times):
                idx = len(self.times) - 1
                
            t_current = self.times[idx]
            pos_current = self.positions[idx]
            des_current = self.desired_positions[idx]
            
            # Update 3D elements – X-frame drone shape
            arms, rotors, body_ls, motor_dots = self._drone_geometry(pos_current, self.attitudes[idx])
            for i, (xs, ys, zs) in enumerate(arms):
                drone_arm_lines3d[i].set_data_3d(xs, ys, zs)
            for i, (xs, ys, zs) in enumerate(rotors):
                drone_rotor_lines3d[i].set_data_3d(xs, ys, zs)
            for i, (xs, ys, zs) in enumerate(body_ls):
                drone_body_lines3d[i].set_data_3d(xs, ys, zs)
            drone_motors3d._offsets3d = ([d[0] for d in motor_dots],
                                         [d[1] for d in motor_dots],
                                         [d[2] for d in motor_dots])
            quad_3d.set_data_3d([pos_current[0]], [pos_current[1]], [pos_current[2]])
            des_3d.set_data_3d([des_current[0]], [des_current[1]], [des_current[2]])
            
            # Update trail (last 30 positions)
            trail_start = max(0, idx - 30)
            trail_line.set_data_3d(self.positions[trail_start:idx+1,0],
                                  self.positions[trail_start:idx+1,1],
                                  self.positions[trail_start:idx+1,2])
            
            # Update attitude plot
            t_data = self.times[:idx+1]
            roll_data = self.attitudes[:idx+1, 0]
            pitch_data = self.attitudes[:idx+1, 1]
            yaw_data = self.attitudes[:idx+1, 2]
            roll_line.set_data(t_data, roll_data)
            pitch_line.set_data(t_data, pitch_data)
            yaw_line.set_data(t_data, yaw_data)
            
            # Update angular rates
            rate_p_data = self.angular_rates[:idx+1, 0]
            rate_q_data = self.angular_rates[:idx+1, 1]
            rate_p_line.set_data(t_data, rate_p_data)
            rate_q_line.set_data(t_data, rate_q_data)
            
            # Update error plot
            err_line.set_data(self.times[:idx+1], self.errors[:idx+1] * 1000)
            
            # Update thrust plot
            thrust_line.set_data(self.times[:idx+1], self.thrusts[:idx+1])
            
            # Update torques plot
            if len(self.torques) > 0:
                torques_arr = np.array(self.torques[:idx+1])
                torque_x_line.set_data(self.times[:idx+1], torques_arr[:, 0])
                torque_y_line.set_data(self.times[:idx+1], torques_arr[:, 1])
                torque_z_line.set_data(self.times[:idx+1], torques_arr[:, 2])
            
            # Update telemetry text
            ax_text.clear()
            ax_text.axis('off')
            
            # Calculate rolling metrics
            window = min(100, idx+1)
            recent_errors = self.errors[max(0, idx-window):idx+1] * 1000
            
            text_str = (f"INTEGRATED DDPG TELEMETRY\n"
                       f"{'='*28}\n"
                       f"Time: {t_current:.1f} / {duration:.0f} s\n"
                       f"Progress: {t_current/duration*100:.0f}%\n\n"
                       f"CURRENT METRICS:\n"
                       f"Position Error: {self.errors[idx]*1000:.1f} mm\n"
                       f"Velocity: {np.linalg.norm(self.velocities[idx]):.2f} m/s\n"
                       f"Altitude: {pos_current[2]:.2f} m\n"
                       f"Thrust: {self.thrusts[idx]:.2f} N\n"
                       f"|τ|: {np.linalg.norm(self.torques[idx]):.3f} Nm\n"
                       f"Roll/Pitch: {self.attitudes[idx,0]:.1f}° / {self.attitudes[idx,1]:.1f}°\n\n"
                       f"ROLLING AVG (last {window} steps):\n"
                       f"Avg Error: {np.mean(recent_errors):.1f} mm\n"
                       f"Max Error: {np.max(recent_errors):.1f} mm")
            
            ax_text.text(0.1, 0.95, text_str, transform=ax_text.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            return (*drone_arm_lines3d, *drone_rotor_lines3d, *drone_body_lines3d,
                    drone_motors3d, quad_3d, des_3d, trail_line, roll_line, pitch_line, yaw_line,
                   rate_p_line, rate_q_line, err_line, thrust_line,
                   torque_x_line, torque_y_line, torque_z_line, ax_text)
        
        # Create animation
        frames = min(len(self.times) // step, total_frames)
        if frames == 0:
            frames = len(self.times) // max(1, step)
            
        anim = FuncAnimation(fig, animate, frames=frames, interval=interval,
                            blit=False, repeat=False)
        
        # Save animation (MP4 via ffmpeg if available, else GIF fallback)
        output_path = os.path.join(self.output_dir, filename)
        self._save_animation(anim, output_path, fps, dpi)
        
        plt.close(fig)
        return anim


def main():
    """Main function to generate videos from trained integrated DDPG agent"""
    
    # Configuration
    # Update this path to your trained model file
    AGENT_PATH = r'weights/agent_best.pth'  # Update with your actual path
    DURATION = 50.0  # seconds
    OUTPUT_DIR = 'integrated_videos'
    
    print("\n" + "="*70)
    print("INTEGRATED DDPG QUADCOPTER VIDEO GENERATION")
    print("="*70)
    print("\nIntegrated DDPG Features:")
    print("  ✓ Single agent (no cascaded architecture)")
    print("  ✓ Direct control: [Ft, τ_x, τ_y, τ_z]")
    print("  ✓ Ornstein-Uhlenbeck exploration")
    print("  ✓ Single critic network")
    print("  ✓ Actor updated every step")
    print("="*70)
    
    # Check if model file exists
    if not os.path.exists(AGENT_PATH):
        print(f"\n⚠ Warning: Integrated DDPG agent not found at: {AGENT_PATH}")
        print("\nLooking for integrated DDPG checkpoints in results directory:")
        
        # Search for integrated controller checkpoints
        found_files = []
        if os.path.exists('results'):
            for d in os.listdir('results'):
                dir_path = f'results/{d}'
                if os.path.isdir(dir_path):
                    for f in os.listdir(dir_path):
                        if 'integrated' in f.lower() and f.endswith('.pth'):
                            found_files.append(f'results/{d}/{f}')
                            print(f"  - results/{d}/{f}")
        
        if not found_files:
            print("  No integrated DDPG checkpoints found in results/")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Create visualizer
    visualizer = IntegratedDDPGVisualizer(
        agent_path=AGENT_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Run simulation
    visualizer.run_simulation(duration=DURATION)
    
    # Generate videos
    print("\n" + "="*70)
    print("GENERATING INTEGRATED DDPG VIDEOS")
    print("="*70)
    
    # Option 1: Simple 3D trajectory animation
    visualizer.create_3d_animation(
        filename='integrated_trajectory_3d.mp4',
        fps=10,
        interval=17,
        dpi=150
    )
    
    # Option 2: Comprehensive dashboard animation
    visualizer.create_dashboard_animation(
        filename='integrated_dashboard.mp4',
        fps=10,
        interval=17,
        dpi=120
    )
    
    print("\n" + "="*70)
    print("INTEGRATED DDPG VIDEO GENERATION COMPLETE!")
    print(f"Videos saved to: {OUTPUT_DIR}/")
    print("="*70)
    
    # Print performance summary
    print("\nINTEGRATED DDPG PERFORMANCE SUMMARY:")
    print(f"  Total steps: {len(visualizer.times)}")
    print(f"  Final error: {visualizer.errors[-1]*1000:.1f} mm")
    print(f"  Mean error: {np.mean(visualizer.errors)*1000:.1f} mm")
    print(f"  Max error: {np.max(visualizer.errors)*1000:.1f} mm")
    print(f"  Std error: {np.std(visualizer.errors)*1000:.1f} mm")
    print(f"  RMSE: {np.sqrt(np.mean(visualizer.errors**2))*1000:.1f} mm")
    print(f"  Success rate (error < 500mm): {np.mean(visualizer.errors*1000 < 500)*100:.1f}%")
    
    # Control effort summary
    print(f"\nCONTROL EFFORT:")
    print(f"  Mean thrust: {np.mean(visualizer.thrusts):.2f} N (hover: {SystemConfig.MASS * SystemConfig.GRAVITY:.2f} N)")
    print(f"  Mean |τ|: {np.mean(np.linalg.norm(visualizer.torques, axis=1)):.3f} Nm")
    print(f"  Max |τ|: {np.max(np.linalg.norm(visualizer.torques, axis=1)):.3f} Nm")


if __name__ == "__main__":
    # Check for matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for video generation
        from matplotlib.animation import FuncAnimation
    except ImportError as e:
        print(f"Error: Required library not found: {e}")
        print("\nPlease install required packages:")
        print("  pip install matplotlib pillow")
        sys.exit(1)
    
    main()