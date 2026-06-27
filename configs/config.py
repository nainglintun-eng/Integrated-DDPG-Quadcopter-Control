"""
Integrated Single-Agent TD3 Config  –  FIXED VERSION
======================================================
Key fixes vs original:
  - max_action = 1.0  (actor outputs normalised [-1,1], env rescales per-channel)
  - Curriculum starts with hover (easier) before full helix
  - Larger buffer, more warmup steps, better LR balance
"""

import numpy as np


class SystemConfig:
    MASS    = 1.0
    GRAVITY = 9.81
    DT      = 0.01   # 100 Hz

    IXX = 0.3
    IYY = 0.4
    IZZ = 0.5
    L   = 0.2        # arm half-length (m)

    # this file is not fully provided to prevent original research.

    # Trajectory parameters (expanding helix)
    A     = 9.81
    B     = 0.01
    OMEGA = 0.2
    VZ    = 1.0


class IntegratedAgentConfig:
    """
    FIX 1: max_action = 1.0  (actor tanh outputs in [-1,+1]).
    Each channel is independently rescaled to its physical range in env.step().
    This prevents torque channels from being permanently saturated.
    """
    STATE_DIM  = 18
    ACTION_DIM = 4

    # FIX 1: normalised action space
    ACTION_SCALE = 1.0   # actor max_action – DO NOT change

    # Per-channel physical limits (used in env.step() for rescaling)
    MAX_THRUST     = SystemConfig.MAX_THRUST
    MIN_THRUST     = SystemConfig.MIN_THRUST
    MAX_TORQUE     = SystemConfig.MAX_TORQUE
    MAX_TORQUE_YAW = SystemConfig.MAX_TORQUE_YAW

    # Max body accel (for obs normalisation of acc_des channel)
    MAX_BODY_ACCEL = 15.0

    # Network
    HIDDEN_DIMS = 256

    # DDPG hyperparameters
    

    # Ornstein-Uhlenbeck noise parameters
    # theta : mean-reversion rate  (0.15 is the classic DDPG default)
    # sigma : noise magnitude      (decayed by trainer across curriculum)
    # ou_dt : matches env DT (0.01 s) for correct OU scaling
  

    # Replay
    BATCH_SIZE   = 256
    BUFFER_SIZE  = 500_000
    WARMUP_STEPS = 3_000

    # Exploration
    EXPLORATION_NOISE = 0.4
    MIN_NOISE         = 0.05

    # Curriculum: start with hover, then scale up
    # Curriculum phases
    # exploration_start / exploration_end now control OU sigma (noise magnitude)
    # rather than a Gaussian noise scale factor.
    CURRICULUM_PHASES = [
        # this file is not fully provided to prevent original research.
    ]


class TrainingConfig:
    SAVE_FREQUENCY           = 500
    LOG_FREQUENCY            = 5
    EVAL_EPISODES            = 5
    KEEP_BEST_ONLY           = True
    MIN_EPISODES_BEFORE_STOP = 100


def get_trajectory_function():
    cfg = SystemConfig()

    def trajectory(t, scale=1.0):
        exp_term = np.exp(-cfg.B * t)
        radius   = cfg.A * (1 - exp_term) * scale
        omega_t  = cfg.OMEGA * t

        x = radius * np.cos(omega_t)
        y = radius * np.sin(omega_t)
        z = cfg.VZ * t * scale

        dR   = cfg.A * cfg.B * exp_term * scale
        vx   = dR * np.cos(omega_t) - radius * cfg.OMEGA * np.sin(omega_t)
        vy   = dR * np.sin(omega_t) + radius * cfg.OMEGA * np.cos(omega_t)
        vz   = cfg.VZ * scale

        d2R  = -cfg.A * cfg.B**2 * exp_term * scale
        ax   = d2R*np.cos(omega_t) - 2*dR*cfg.OMEGA*np.sin(omega_t) - radius*cfg.OMEGA**2*np.cos(omega_t)
        ay   = d2R*np.sin(omega_t) + 2*dR*cfg.OMEGA*np.cos(omega_t) - radius*cfg.OMEGA**2*np.sin(omega_t)
        az   = 0.0

        return np.array([x,y,z]), np.array([vx,vy,vz]), np.array([ax,ay,az])

    return trajectory
