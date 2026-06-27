"""
Integrated Single-Agent Environment  –  FIXED VERSION
=======================================================
Root causes fixed (see FIXES below):

  FIX 1 – Per-channel action normalisation
    Actor outputs a(4) in [-1, +1]  (tanh, max_action=1).
    step() rescales each channel to its own physical range before clamping:
        a[0] → [MIN_THRUST, MAX_THRUST]       (mapped from [-1,1])
        a[1] → [-MAX_TORQUE, +MAX_TORQUE]
        a[2] → [-MAX_TORQUE, +MAX_TORQUE]
        a[3] → [-MAX_TORQUE_YAW, +MAX_TORQUE_YAW]
    This lets the actor explore the full torque range without saturation.

  FIX 2 – Reward: removed misleading precision bonus.
    Replaced with a strong, dense exponential tracking reward that
    clearly distinguishes good from bad policy at every step.

  FIX 3 – Reward: hover survival bonus.
    A positive bonus is given proportionally to how close Ft is to hover
    thrust (m*g), incentivising the agent to stay airborne.

  FIX 4 – Reward: penalty for crashing / going out of bounds (terminal).
    A large negative reward on termination gives a clear failure signal.

State ordering  (MATLAB-matched):
    [x, y, z, u, v, w, phi, theta, psi, p, q, r]
     0  1  2  3  4  5   6    7     8   9  10  11

Observation  (18-dim):
    pos_error / MAX_POSITION   (3)
    vel / MAX_VELOCITY         (3)
    att                        (3)  [phi, theta, psi]
    rates / 10                 (3)
    vel_des / MAX_VELOCITY     (3)
    acc_des / MAX_BODY_ACCEL   (3)

Action  (4-dim, actor outputs in [-1, +1]):
    step() maps to physical range per channel (FIX 1).
"""

import numpy as np
import gym
from gym import spaces
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.dynamics import QuadcopterDynamics
from configs.config import SystemConfig, IntegratedAgentConfig, get_trajectory_function


class IntegratedEnv(gym.Env):

    def __init__(self, config=None):
        super().__init__()

        self.sys_cfg = SystemConfig()
        self.cfg     = config if config is not None else IntegratedAgentConfig()

        self.dynamics = QuadcopterDynamics(
            mass=self.sys_cfg.MASS, gravity=self.sys_cfg.GRAVITY, dt=self.sys_cfg.DT)

        self.get_trajectory = get_trajectory_function()

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.STATE_DIM,), dtype=np.float32)

        # FIX 1: Actor outputs in [-1, +1], max_action=1.
        # Per-channel rescaling happens in step().
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.cfg.ACTION_DIM,), dtype=np.float32)

        self.state      = None
        self.time       = 0.0
        self.steps      = 0
        self.max_steps  = self.sys_cfg.MAX_STEPS

        self.trajectory_scale = 1.0
        self.wind_std         = [0.0, 0.0, 0.0]
        self.start_radius     = 0.0
        self.prev_action      = None

        # History for visualisation
        self.trajectory_history         = []
        self.desired_trajectory_history = []
        self.attitude_history           = []
        self.control_history            = []

    # ---------------------------------------------------------------
    #  Curriculum setters
    # ---------------------------------------------------------------
    def set_trajectory_scale(self, scale): self.trajectory_scale = scale
    def set_start_radius(self, radius):    self.start_radius = radius

    def set_wind(self, std):
        self.wind_std = list(std) if hasattr(std, '__iter__') else [std, std, std*0.6]
        self.dynamics.set_wind(
            enabled=any(s > 0 for s in self.wind_std),
            std=self.wind_std, mean=[0,0,0],
            gust_prob=self.sys_cfg.WIND_GUST_PROB,
            gust_mag=self.sys_cfg.WIND_GUST_MAG)

    # ---------------------------------------------------------------
    #  Reset
    # ---------------------------------------------------------------
    def reset(self, initial_pos=None):
        self.time  = 0.0
        self.steps = 0

        if initial_pos is None:
            if self.sys_cfg.RANDOM_START and self.start_radius > 0:
                angle       = np.random.uniform(0, 2*np.pi)
                radius      = np.random.uniform(0.3, self.start_radius)
                z           = np.random.uniform(0, min(1.5, self.start_radius/2))
                initial_pos = np.array([radius*np.cos(angle),
                                        radius*np.sin(angle), z])
            else:
                initial_pos = np.random.uniform(-0.05, 0.05, 3)

        self.state       = np.zeros(12)
        self.state[0:3]  = initial_pos
        self.state[2]    = max(0.0, initial_pos[2])
        self.state[3:6]  = np.random.uniform(-0.05, 0.05, 3)
        self.state[6]    = np.random.uniform(-0.03, 0.03)
        self.state[7]    = np.random.uniform(-0.03, 0.03)
        self.state[8]    = np.random.uniform(-0.05, 0.05)

        self.set_wind(self.wind_std)
        self.prev_action = None
        self.trajectory_history         = []
        self.desired_trajectory_history = []
        self.attitude_history           = []
        self.control_history            = []

        return self._get_observation()

    # ---------------------------------------------------------------
    #  Step
    # ---------------------------------------------------------------
    def step(self, action):
        """
        action: (4,) in [-1, +1]  (actor tanh output, max_action=1)

        FIX 1 – per-channel physical rescaling:
            Ft    = MIN_T + (a[0]+1)/2 * (MAX_T - MIN_T)
            tau_x = a[1] * MAX_TORQUE
            tau_y = a[2] * MAX_TORQUE
            tau_z = a[3] * MAX_TORQUE_YAW
        """
        action = np.clip(np.array(action, dtype=np.float64), -1.0, 1.0)

        # Map each channel to its own physical range
        Ft    = self.cfg.MIN_THRUST + (action[0] + 1.0) * 0.5 * \
                (self.cfg.MAX_THRUST - self.cfg.MIN_THRUST)
        tau_x = action[1] * self.cfg.MAX_TORQUE
        tau_y = action[2] * self.cfg.MAX_TORQUE
        tau_z = action[3] * self.cfg.MAX_TORQUE_YAW

        torques = np.array([tau_x, tau_y, tau_z])

        self.state = self.dynamics.rk4_step(self.state, Ft, torques)
        self.time  += self.sys_cfg.DT
        self.steps += 1

        # History
        self.trajectory_history.append(self.state[0:3].copy())
        pos_des, _, _ = self.get_trajectory(self.time, self.trajectory_scale)
        self.desired_trajectory_history.append(pos_des.copy())
        self.attitude_history.append(np.rad2deg(self.state[6:9].copy()))
        physical_u = np.array([Ft, tau_x, tau_y, tau_z])
        self.control_history.append(physical_u.copy())

        # Reward
        reward, info = self._compute_reward(physical_u, action)
        done, term   = self._check_done()

        # FIX 4: terminal penalty
        if done and term in ('crash', 'out_of_bounds', 'altitude_limit',
                             'excessive_velocity', 'excessive_rate'):
            reward -= 50.0

        info['termination'] = term
        info['Ft']     = Ft
        info['torques'] = torques

        self.prev_action = action.copy()
        return self._get_observation(), reward, done, info

    # ---------------------------------------------------------------
    #  Observation
    # ---------------------------------------------------------------
    def _get_observation(self):
        pos   = self.state[0:3]
        vel   = self.state[3:6]
        att   = self.state[6:9]
        rates = self.state[9:12]

        pos_des, vel_des, acc_des = self.get_trajectory(
            self.time, self.trajectory_scale)

        rel_pos  = (pos - pos_des) / self.sys_cfg.MAX_POSITION
        norm_vel = vel / self.sys_cfg.MAX_VELOCITY

        obs = np.concatenate([
            rel_pos, norm_vel, att, rates / 10.0,
            vel_des / self.sys_cfg.MAX_VELOCITY,
            acc_des / self.cfg.MAX_BODY_ACCEL
        ])
        return obs.astype(np.float32)

    # ---------------------------------------------------------------
    #  Reward  (FIX 2 + FIX 3)
    # ---------------------------------------------------------------
    def _compute_reward(self, physical_u, norm_action):
        """
        Dense, well-scaled reward that clearly distinguishes hover from random.

        Components:
          1. Exponential position reward  – peak at 0 error, decays sharply
          2. Velocity penalty             – penalise excess speed
          3. Attitude penalty             – penalise large roll/pitch
          4. Rate penalty                 – penalise spinning
          5. Hover survival bonus         – reward thrust close to m*g
          6. Smoothness penalty           – penalise jitter
          7. Termination penalty          – applied in step() on crash
        """
        pos   = self.state[0:3]
        vel   = self.state[3:6]
        att   = self.state[6:9]
        rates = self.state[9:12]

        pos_des, vel_des, _ = self.get_trajectory(self.time, self.trajectory_scale)
        Ft = physical_u[0]

        pos_err = np.linalg.norm(pos - pos_des)
        vel_err = np.linalg.norm(vel - vel_des)

        # 1. Exponential position reward  (peak=+10, decays to ~0 at 3 m)
        pos_r = 10.0 * np.exp(-2.0 * pos_err)

        # 2. Velocity penalty
        vel_p = -0.5 * (vel_err / self.sys_cfg.MAX_VELOCITY) ** 2

        # 3. Attitude penalty (roll / pitch, normalised by 15°)
        att_norm = np.deg2rad(15.0)
        att_p = -1.0 * np.sum(np.square(att[0:2] / att_norm))

        # 4. Angular rate penalty
        rate_p = -0.3 * np.sum(np.square(rates / 5.0))

        # 5. Hover survival bonus: reward Ft close to m*g
        hover_thrust = self.sys_cfg.MASS * self.sys_cfg.GRAVITY
        thrust_err   = abs(Ft - hover_thrust) / hover_thrust
        survival_r   = 1.0 * np.exp(-3.0 * thrust_err)

        # 6. Smoothness penalty on normalised action
        smooth_p = 0.0
        if self.prev_action is not None:
            delta    = norm_action - self.prev_action
            smooth_p = -0.05 * np.sum(np.square(delta))

        total = pos_r + vel_p + att_p + rate_p + survival_r + smooth_p

        info = {
            'pos_error':   pos_err * 1000,   # mm
            'vel_error':   vel_err,
            'pos_error_m': pos_err,
            'reward_components': {
                'position':   pos_r,
                'velocity':   vel_p,
                'attitude':   att_p,
                'rate':       rate_p,
                'survival':   survival_r,
                'smoothness': smooth_p,
                'total':      total
            }
        }
        return total, info

    # ---------------------------------------------------------------
    #  Termination
    # ---------------------------------------------------------------
    def _check_done(self):
        pos   = self.state[0:3]
        att   = self.state[6:9]
        vel   = self.state[3:6]
        rates = self.state[9:12]

        if (np.abs(att[0]) > self.sys_cfg.CRASH_ANGLE or
                np.abs(att[1]) > self.sys_cfg.CRASH_ANGLE):
            return True, 'crash'
        if (np.abs(pos[0]) > self.sys_cfg.MAX_POSITION or
                np.abs(pos[1]) > self.sys_cfg.MAX_POSITION):
            return True, 'out_of_bounds'
        if pos[2] < self.sys_cfg.MIN_ALTITUDE or pos[2] > self.sys_cfg.MAX_ALTITUDE:
            return True, 'altitude_limit'
        if np.linalg.norm(vel) > self.sys_cfg.MAX_VELOCITY:
            return True, 'excessive_velocity'
        if np.max(np.abs(rates)) > self.sys_cfg.MAX_ANGULAR_RATE:
            return True, 'excessive_rate'
        if self.steps >= self.max_steps:
            return True, 'max_steps'
        return False, None

    def render(self, mode='human'):
        pass
