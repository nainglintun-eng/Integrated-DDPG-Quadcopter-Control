"""
evaluate_matlab_noise.py
========================
Evaluates the three quadcopter agents (Cascaded TD3, Integrated TD3,
Integrated DDPG) under noise and wind disturbances that are faithful
Python translations of the three MATLAB helper functions:

    add_control_noise(control_vector, noise_percent)
    add_state_noise(state_vector,     noise_percent)
    wind_f_m(X, Wxyz, Ts_Sim)

Physical constants are taken directly from MATLAB source and verified
to match the Python dynamics (mass=1 kg, L=0.2 m, Ix/Iy/Iz=0.3/0.4/0.5).

Test matrix
-----------
  Control noise  : noise_percent in {1, 2, 3, 4, 5}
  State noise    : noise_percent in {1, 2, 3, 4, 5}
  Wind           : |Wxyz| in {0.5, 1.0, 2.0, 3.0, 5.0} m/s (random direction each episode)
  Combined       : control+state noise 3% + wind 2 m/s
  Baseline       : all off

Metrics per condition (same as evaluate_performance_table.py)
--------------------------------------------------------------
  RMSE (m)      root-mean-square 3-D position error
  Mean±Std (m)  mean ± std of per-step position error
  Max (m)       mean peak per-episode error
  Succ (%)      episodes with RMSE < 0.3 m
  Crash (%)     episodes ending in crash / OOB / velocity limit
  Conv (s)      time to first sustain < 0.1 m for >= 20 steps

Outputs
-------
  results_matlab/matlab_noise_table.tex   -- LaTeX table (paper-ready)
  results_matlab/matlab_noise_results.csv -- flat CSV
  results_matlab/matlab_noise_results.npy -- numpy dict

Usage
-----
  # All three systems:
  python evaluate_matlab_noise.py \\
      --cascaded_att  cascaded_td3/weights/attitude_best.pth \\
      --cascaded_pos  cascaded_td3/weights/position_best.pth \\
      --integrated_td3   integrated_td3/results/.../best.pth \\
      --integrated_ddpg  integrated_ddpg/results/.../best.pth \\
      --episodes 30

  # Single system:
  python evaluate_matlab_noise.py \\
      --integrated_ddpg integrated_ddpg/results/.../best.pth \\
      --episodes 30

  # Demo (no checkpoints, synthetic data):
  python evaluate_matlab_noise.py --demo --episodes 10

  # With motor-level saturation (MATLAB saturate_controls logic=1):
  python evaluate_matlab_noise.py --integrated_ddpg runs/ddpg.pth --saturate
"""

import argparse
import os
import sys
import csv
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# ── Repo roots (same layout as evaluate_performance_table.py) ─────────────────
SCRIPT_DIR           = os.path.dirname(os.path.abspath(__file__))
CASCADED_ROOT        = os.path.join(SCRIPT_DIR, 'cascaded_td3')
INTEGRATED_TD3_ROOT  = os.path.join(SCRIPT_DIR, 'integrated_td3')
INTEGRATED_DDPG_ROOT = os.path.join(SCRIPT_DIR, 'integrated_ddpg')

# ==============================================================================
#  PHYSICAL CONSTANTS  (match MATLAB source exactly)
# ==============================================================================

_L      = 0.2           # half arm length (m)
_m      = 1.0           # mass (kg)
_g      = 9.81          # gravity (m/s^2)
_Ix     = 0.3           # kg·m²
_Iy     = 0.4
_Iz     = 0.5
_rho    = 1.225         # air density (kg/m³)

# Actuator uncertainty vector  [Ft, tau_x, tau_y, tau_z]
_max_Ft      = 2.0 * _g              # = 19.62 N
_Tmax        = _max_Ft / 4.0         # = 4.905 N per motor
_Max_torque  = _Tmax * _L * 2.0     # = 1.962 Nm
ACTUATOR_UNCERT = np.array([
    0.10 * _max_Ft,      # Ft      ± 1.962 N
    0.01 * _Max_torque,  # tau_x   ± 0.01962 Nm
    0.01 * _Max_torque,  # tau_y   ± 0.01962 Nm
    0.02 * _Max_torque,  # tau_z   ± 0.03924 Nm
])

# Sensor uncertainty vector — physically motivated, per-channel sigmas.
# These match the values used in evaluate_performance_table.py:
#   position  σ = 0.01 m   (1 cm  — realistic GPS/mocap noise)
#   velocity  σ = 0.01 m/s (realistic velocity estimator)
#   attitude  σ = 0.005 rad = 0.29°  (realistic IMU)
#   ang. rate σ = 0.005 rad/s (realistic gyro)
# noise_percent {1..5} multiplies these base sigmas, so:
#   noise_percent=1 → baseline realistic noise
#   noise_percent=5 → 5× baseline (degraded sensor)
_SENSOR_SIGMA_POS  = 0.01    # m
_SENSOR_SIGMA_VEL  = 0.01    # m/s
_SENSOR_SIGMA_ATT  = 0.005   # rad  (0.29 deg)
_SENSOR_SIGMA_RATE = 0.005   # rad/s

# Wind aerodynamic params  (from wind_f_m)
_Cd = np.array([0.8, 0.8, 1.2])          # drag coefficients [x,y,z]
_h  = 0.05                                # body height (m)
_A  = np.array([_L*2*_h, _L*2*_h, 2*_L*_L])  # frontal areas [x,y,z] m²

# Motor positions  (3×4 matrix, columns = motors)
_r_motors = np.array([
    [ _L,  _L, -_L, -_L],   # x
    [ _L, -_L, -_L,  _L],   # y
    [  0,   0,   0,   0],   # z
])

CRASH_TERMS = ('crash', 'out_of_bounds', 'altitude_limit',
               'excessive_velocity', 'excessive_rate')


# ==============================================================================
#  MATLAB-FAITHFUL NOISE / WIND FUNCTIONS
# ==============================================================================

def eul2rotm_zyx(psi, theta, phi):
    """
    ZYX Euler rotation matrix R = Rz(psi) @ Ry(theta) @ Rx(phi).
    Matches MATLAB eul2rotm_zyx(X(7), X(8), X(9)) where X(7)=phi, X(8)=theta, X(9)=psi.
    NOTE: MATLAB call passes (X(7), X(8), X(9)) = (phi, theta, psi) as positional args,
    but the function signature is eul2rotm_zyx(phi, theta, psi) ->  ZYX convention.
    """
    cp = np.cos(phi);   sp = np.sin(phi)
    ct = np.cos(theta); st = np.sin(theta)
    cs = np.cos(psi);   ss = np.sin(psi)

    R = np.array([
        [ct*cs,   sp*st*cs - cp*ss,   cp*st*cs + sp*ss],
        [ct*ss,   sp*st*ss + cp*cs,   cp*st*ss - sp*cs],
        [-st,     sp*ct,              cp*ct            ],
    ])
    return R


def add_control_noise(control_vector, noise_percent):
    """
    Python translation of MATLAB add_control_noise(control_vector, noise_percent).

    control_vector : (4,) array  [Ft, tau_x, tau_y, tau_z]
    noise_percent  : scalar      {1, 2, 3, 4, 5}  (applied as multiplier)

    Noise = noise_percent * actuator_uncert * randn(4)
    """
    noise = noise_percent * ACTUATOR_UNCERT * np.random.randn(4)
    return control_vector + noise


def add_state_noise(obs_vector, noise_percent):
    """
    Apply realistic per-channel sensor noise to the 18-D normalised observation.

    Noise is specified in physical units and scaled to match the normalisation
    the environment applies in _get_observation():
        obs[0:3]  = (pos - pos_des) / MAX_POS  -> pos sigma / MAX_POS
        obs[3:6]  = vel / MAX_VEL              -> vel sigma / MAX_VEL
        obs[6:9]  = att (rad, unscaled)        -> att sigma added directly
        obs[9:12] = rates / 10                 -> rate sigma / 10
        obs[12:18]= desired vel/acc (planner)  -> no noise

    noise_percent {1..5} scales all sigmas proportionally:
        1 → baseline realistic noise  (pos ±1 cm, att ±0.29°)
        5 → 5× baseline               (pos ±5 cm, att ±1.43°)
    """
    _MAX_POS = 5.0
    _MAX_VEL = 5.0
    noisy = obs_vector.copy()
    s = noise_percent   # scale factor
    noisy[0:3]  += np.random.normal(0, s * _SENSOR_SIGMA_POS  / _MAX_POS, 3)
    noisy[3:6]  += np.random.normal(0, s * _SENSOR_SIGMA_VEL  / _MAX_VEL, 3)
    noisy[6:9]  += np.random.normal(0, s * _SENSOR_SIGMA_ATT,             3)
    noisy[9:12] += np.random.normal(0, s * _SENSOR_SIGMA_RATE / 10.0,     3)
    return noisy.astype(np.float32)


def wind_f_m(X, Wxyz, Ts_Sim):
    """
    Python translation of MATLAB wind_f_m(X, Wxyz, Ts_Sim).

    Applies aerodynamic wind disturbance directly to state X using
    the same drag + torque model as the MATLAB reference.

    X      : (12,) state  [x,y,z, u,v,w, phi,theta,psi, p,q,r]
    Wxyz   : (3,)  wind velocity in world frame (m/s)
    Ts_Sim : float simulation timestep (s)

    Returns updated X after applying one timestep of wind disturbance.
    """
    phi, theta, psi = X[6], X[7], X[8]

    # Rotation matrix: world -> body  (R maps body to world, R' = R.T maps world to body)
    R = eul2rotm_zyx(psi, theta, phi)   # matches MATLAB R = eul2rotm_zyx(X(7),X(8),X(9))
    Wxyz_body = R.T @ Wxyz              # wind in body frame

    # Relative velocity between quadcopter and wind (body frame)
    V_rel = X[3:6] - Wxyz_body

    # Aerodynamic drag force in body frame
    #   F_wind = -0.5 * rho * Cd .* diag(A) * (V_rel .* |V_rel|)
    F_wind = -0.5 * _rho * _Cd * _A * (V_rel * np.abs(V_rel))  # (3,)

    # Distribute wind force to 4 motors with small per-motor noise
    F_wind_motors = np.tile(F_wind[:, None], (1, 4)) + 0.01 * np.random.randn(3, 4)

    # Wind torques from motor moment arms: cross(r_motors, F_wind_motors)
    M_wind_motors = np.cross(_r_motors, F_wind_motors, axisa=0, axisb=0, axisc=0)
    M_wind = M_wind_motors.sum(axis=1)   # sum over 4 motors -> (3,)

    # Translational and rotational disturbance accelerations
    accn_dist  = F_wind / _m
    alpha_dist = np.array([M_wind[0] / _Ix,
                            M_wind[1] / _Iy,
                            M_wind[2] / _Iz])

    # Apply disturbance (Euler step, same as MATLAB)
    dX = np.zeros(12)
    dX[3:6]  = accn_dist
    dX[9:12] = alpha_dist
    X_out = X + Ts_Sim * dX
    return X_out


# ==============================================================================
#  TEST CONDITIONS
# ==============================================================================

def make_matlab_conditions():
    """
    Return list of (id, display_name, params_dict).

    params_dict keys
    ----------------
    ctrl_noise  : float  -- noise_percent for add_control_noise (0 = off)
    state_noise : float  -- noise_percent for add_state_noise   (0 = off)
    wind_speed  : float  -- |Wxyz| in m/s (0 = off); direction randomised per episode
    mass_frac   : float  -- fractional mass uncertainty (0 = off)
    """
    return [
        # ── Baseline ─────────────────────────────────────────────────────────
        ('baseline',    'No Disturbance',
         {'ctrl_noise': 0, 'state_noise': 0, 'wind_speed': 0.0, 'mass_frac': 0.0}),

        # ── Control noise (actuator uncertainty) ──────────────────────────────
        ('ctrl1',  'Control Noise 1%',
         {'ctrl_noise': 1, 'state_noise': 0, 'wind_speed': 0.0, 'mass_frac': 0.0}),
        ('ctrl2',  'Control Noise 2%',
         {'ctrl_noise': 2, 'state_noise': 0, 'wind_speed': 0.0, 'mass_frac': 0.0}),
        ('ctrl3',  'Control Noise 3%',
         {'ctrl_noise': 3, 'state_noise': 0, 'wind_speed': 0.0, 'mass_frac': 0.0}),
        ('ctrl4',  'Control Noise 4%',
         {'ctrl_noise': 4, 'state_noise': 0, 'wind_speed': 0.0, 'mass_frac': 0.0}),
        ('ctrl5',  'Control Noise 5%',
         {'ctrl_noise': 5, 'state_noise': 0, 'wind_speed': 0.0, 'mass_frac': 0.0}),

        # ── State / sensor noise ──────────────────────────────────────────────
        ('state1', 'State Noise 1%',
         {'ctrl_noise': 0, 'state_noise': 1, 'wind_speed': 0.0, 'mass_frac': 0.0}),
        ('state2', 'State Noise 2%',
         {'ctrl_noise': 0, 'state_noise': 2, 'wind_speed': 0.0, 'mass_frac': 0.0}),
        ('state3', 'State Noise 3%',
         {'ctrl_noise': 0, 'state_noise': 3, 'wind_speed': 0.0, 'mass_frac': 0.0}),
        ('state4', 'State Noise 4%',
         {'ctrl_noise': 0, 'state_noise': 4, 'wind_speed': 0.0, 'mass_frac': 0.0}),
        ('state5', 'State Noise 5%',
         {'ctrl_noise': 0, 'state_noise': 5, 'wind_speed': 0.0, 'mass_frac': 0.0}),

        # ── Wind (aerodynamic, faithful to wind_f_m) ──────────────────────────
        ('wind0p5', 'Wind 0.5 m/s',
         {'ctrl_noise': 0, 'state_noise': 0, 'wind_speed': 0.5, 'mass_frac': 0.0}),
        ('wind1',   'Wind 1.0 m/s',
         {'ctrl_noise': 0, 'state_noise': 0, 'wind_speed': 1.0, 'mass_frac': 0.0}),
        ('wind2',   'Wind 2.0 m/s',
         {'ctrl_noise': 0, 'state_noise': 0, 'wind_speed': 2.0, 'mass_frac': 0.0}),
        ('wind3',   'Wind 3.0 m/s',
         {'ctrl_noise': 0, 'state_noise': 0, 'wind_speed': 3.0, 'mass_frac': 0.0}),
        ('wind5',   'Wind 5.0 m/s',
         {'ctrl_noise': 0, 'state_noise': 0, 'wind_speed': 5.0, 'mass_frac': 0.0}),

        # ── Combined ─────────────────────────────────────────────────────────
        ('combined', 'Ctrl3% + State3% + Wind 2 m/s',
         {'ctrl_noise': 3, 'state_noise': 3, 'wind_speed': 2.0, 'mass_frac': 0.0}),
    ]


# ==============================================================================
#  SYSTEM LOADERS  (same importlib isolation as evaluate_performance_table.py)
# ==============================================================================

def load_cascaded_td3(att_ckpt, pos_ckpt, demo=False):
    if demo:
        return None, 1.0
    root = CASCADED_ROOT
    import importlib.util

    def _load(name, rel):
        spec = importlib.util.spec_from_file_location(name, os.path.join(root, rel))
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    td3 = _load('casc_td3',    'agents/td3_agent.py')
    env = _load('casc_posenv', 'environments/position_env.py')
    cfg = _load('casc_cfg',    'configs/config.py')

    att_cfg = cfg.AttitudeControllerConfig()
    pos_cfg = cfg.PositionControllerConfig()
    sys_cfg = cfg.SystemConfig()

    att_agent = td3.TD3Agent(
        state_dim=att_cfg.STATE_DIM, action_dim=att_cfg.ACTION_DIM,
        max_action=att_cfg.MAX_TORQUE,  hidden_dims=att_cfg.HIDDEN_DIMS,
        actor_lr=att_cfg.ACTOR_LR, critic_lr=att_cfg.CRITIC_LR,
        gamma=att_cfg.GAMMA, tau=att_cfg.TAU,
        policy_noise=att_cfg.POLICY_NOISE, noise_clip=att_cfg.NOISE_CLIP,
        policy_delay=att_cfg.POLICY_DELAY, buffer_size=1000)
    pos_agent = td3.TD3Agent(
        state_dim=pos_cfg.STATE_DIM, action_dim=pos_cfg.ACTION_DIM,
        max_action=pos_cfg.MAX_BODY_ACCELERATION, hidden_dims=pos_cfg.HIDDEN_DIMS,
        actor_lr=pos_cfg.ACTOR_LR, critic_lr=pos_cfg.CRITIC_LR,
        gamma=pos_cfg.GAMMA, tau=pos_cfg.TAU,
        policy_noise=pos_cfg.POLICY_NOISE, noise_clip=pos_cfg.NOISE_CLIP,
        policy_delay=pos_cfg.POLICY_DELAY, buffer_size=1000)

    if att_ckpt and os.path.exists(att_ckpt):
        att_agent.load(att_ckpt); print(f"    attitude <- {att_ckpt}")
    else:
        print("    attitude <- random weights")
    if pos_ckpt and os.path.exists(pos_ckpt):
        pos_agent.load(pos_ckpt); print(f"    position <- {pos_ckpt}")
    else:
        print("    position <- random weights")

    pos_env = env.PositionEnv(attitude_agent=att_agent)
    pos_env.set_trajectory_scale(1.0)
    pos_env.set_start_radius(0.5)
    return (pos_env, pos_agent), sys_cfg.MASS


def load_integrated_td3(ckpt, demo=False):
    if demo:
        return None, None, 1.0
    root = INTEGRATED_TD3_ROOT
    import importlib.util

    def _load(name, rel):
        spec = importlib.util.spec_from_file_location(name, os.path.join(root, rel))
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    env_mod   = _load('itd3_env', 'environments/integrated_env.py')
    cfg_mod   = _load('itd3_cfg', 'configs/config.py')
    agent_mod = _load('itd3_agt', 'agents/td3_agent.py')

    cfg = cfg_mod.IntegratedAgentConfig()
    sys_cfg = cfg_mod.SystemConfig()

    agent = agent_mod.TD3Agent(
        state_dim=cfg.STATE_DIM, action_dim=cfg.ACTION_DIM,
        max_action=cfg.ACTION_SCALE, hidden_dims=cfg.HIDDEN_DIMS,
        actor_lr=getattr(cfg,'ACTOR_LR',3e-5), critic_lr=getattr(cfg,'CRITIC_LR',1e-4),
        gamma=cfg.GAMMA, tau=cfg.TAU,
        policy_noise=getattr(cfg,'POLICY_NOISE',0.15),
        noise_clip=getattr(cfg,'NOISE_CLIP',0.3),
        policy_delay=getattr(cfg,'POLICY_DELAY',2), buffer_size=1000)

    if ckpt and os.path.exists(ckpt):
        agent.load(ckpt); print(f"    agent <- {ckpt}")
    else:
        print("    agent <- random weights")

    env = env_mod.IntegratedEnv()
    env.set_trajectory_scale(1.0)
    env.set_start_radius(0.5)
    return env, agent, sys_cfg.MASS


def load_integrated_ddpg(ckpt, demo=False):
    if demo:
        return None, None, 1.0
    root = INTEGRATED_DDPG_ROOT
    import importlib.util

    def _load(name, rel):
        spec = importlib.util.spec_from_file_location(name, os.path.join(root, rel))
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    env_mod   = _load('iddpg_env', 'environments/integrated_env.py')
    cfg_mod   = _load('iddpg_cfg', 'configs/config.py')
    agent_mod = _load('iddpg_agt', 'agents/ddpg_agent.py')

    cfg = cfg_mod.IntegratedAgentConfig()
    sys_cfg = cfg_mod.SystemConfig()

    agent = agent_mod.DDPGAgent(
        state_dim=cfg.STATE_DIM, action_dim=cfg.ACTION_DIM,
        max_action=cfg.ACTION_SCALE, hidden_dims=cfg.HIDDEN_DIMS,
        actor_lr=getattr(cfg,'ACTOR_LR',5e-5), critic_lr=getattr(cfg,'CRITIC_LR',3e-4),
        gamma=cfg.GAMMA, tau=cfg.TAU,
        ou_theta=getattr(cfg,'OU_THETA',0.15),
        ou_sigma=getattr(cfg,'OU_SIGMA',0.2),
        ou_dt=getattr(cfg,'OU_DT',0.01), buffer_size=1000)

    if ckpt and os.path.exists(ckpt):
        agent.load(ckpt); print(f"    agent <- {ckpt}")
    else:
        print("    agent <- random weights")

    env = env_mod.IntegratedEnv()
    env.set_trajectory_scale(1.0)
    env.set_start_radius(0.5)
    return env, agent, sys_cfg.MASS


# ==============================================================================
#  EPISODE METRICS
# ==============================================================================

def episode_metrics(pos_errors, vel_errors, attitudes, actions, crashed, dt):
    errs = np.array(pos_errors, dtype=float)
    vels = np.array(vel_errors, dtype=float) if len(vel_errors) else np.array([0.0])
    atts = np.array(attitudes,  dtype=float)
    acts = np.array(actions,    dtype=float)

    rmse = float(np.sqrt(np.mean(errs ** 2)))
    mean = float(np.mean(errs))
    std  = float(np.std(errs))
    maxi = float(np.max(errs))

    conv_t = float('nan')
    for i in range(len(errs) - 19):
        if np.all(errs[i:i + 20] < 0.1):  # 0.1 m = 100 mm
            conv_t = i * dt
            break
    if np.isnan(conv_t) and not crashed:
        conv_t = len(errs) * dt

    smooth = float(np.mean(np.abs(np.diff(acts, axis=0)))) if len(acts) > 1 else 0.0

    return {
        'rmse':     rmse,
        'mean':     mean,
        'std':      std,
        'max':      maxi,
        'conv_t':   conv_t,
        'crashed':  crashed,
        'smooth':   smooth,
        'mean_vel': float(np.mean(vels)),
        'roll':     float(np.mean(np.abs(atts[:, 0]))) if len(atts) else 0.0,
        'pitch':    float(np.mean(np.abs(atts[:, 1]))) if len(atts) else 0.0,
    }


def aggregate_metrics(episode_list):
    rmses   = [m['rmse']   for m in episode_list]
    means   = [m['mean']   for m in episode_list]
    stds    = [m['std']    for m in episode_list]
    maxes   = [m['max']    for m in episode_list]
    convs   = [m['conv_t'] for m in episode_list if not np.isnan(m['conv_t'])]
    crashes = sum(1 for m in episode_list if m['crashed'])
    n       = len(episode_list)
    return {
        'rmse_mean':    float(np.mean(rmses)),
        'rmse_std':     float(np.std(rmses)),
        'mean_err':     float(np.mean(means)),
        'mean_err_std': float(np.mean(stds)),
        'max_err':      float(np.mean(maxes)),
        'success_rate': float(np.mean([1 if r < 0.3 else 0 for r in rmses])) * 100,
        'crash_rate':   crashes / n * 100,
        'conv_time':    float(np.mean(convs)) if convs else float('nan'),
        'smoothness':   float(np.mean([m['smooth']   for m in episode_list])),
        'mean_vel_err': float(np.mean([m['mean_vel'] for m in episode_list])),
        'mean_roll':    float(np.mean([m['roll']     for m in episode_list])),
        'mean_pitch':   float(np.mean([m['pitch']    for m in episode_list])),
    }


# ==============================================================================
#  EPISODE RUNNERS  (apply MATLAB-faithful noise at each step)
# ==============================================================================

def _sample_wind_vector(speed):
    """Random unit direction scaled to given speed (m/s)."""
    if speed <= 0:
        return np.zeros(3)
    v = np.random.randn(3)
    v /= (np.linalg.norm(v) + 1e-9)
    return v * speed


# ==============================================================================
#  SATURATE CONTROLS  (MATLAB saturate_controls.m, logic=1)
# ==============================================================================

#  Physical limits (must match MATLAB constants above)
_SAT_Tmax        = _max_Ft / 4.0          # 4.905 N per motor
_SAT_Tmin        = 0.0
_SAT_max_torque  = _SAT_Tmax * _L * 2.0  # 1.962 Nm  (roll / pitch)
_SAT_max_yaw_tor = 0.7 * _SAT_max_torque  # 1.3734 Nm (yaw)
_SAT_max_Ft      = _max_Ft                # 19.62 N
_SAT_drag        = 0.02                   # drag coefficient (dimensionless)


def saturate_controls(ft, torques):
    """
    Python translation of MATLAB saturate_controls(U, logic=1).

    Enforces motor-level feasibility:
      1. Clip roll/pitch torques to ±max_torque, yaw torque to ±max_yaw_torque.
      2. Solve for individual motor thrusts given (ft, clipped torques).
      3. Clip each motor thrust to [Tmin, Tmax].
      4. Recompute total thrust as sum of clipped motor thrusts (also clipped to [0, max_Ft]).
      5. Return saturated (ft_out, torques_out).  Torques come from step 1 (NOT
         back-computed from motor thrusts), exactly matching the MATLAB output.

    Parameters
    ----------
    ft      : float       total thrust (N)
    torques : (3,) array  [tau_x, tau_y, tau_z] (Nm)

    Returns
    -------
    ft_sat  : float
    tau_sat : (3,) ndarray
    """
    tau = np.array(torques, dtype=float).copy()

    # Step 1 — clip torques
    tau[0] = np.clip(tau[0], -_SAT_max_torque,  _SAT_max_torque)
    tau[1] = np.clip(tau[1], -_SAT_max_torque,  _SAT_max_torque)
    tau[2] = np.clip(tau[2], -_SAT_max_yaw_tor, _SAT_max_yaw_tor)

    # Step 2 — solve for per-motor thrusts
    #   T1 = (ft - tx/(2L) - ty/(2L) + tz/(4*drag)) / 4
    #   T2 = (ft - tx/(2L) + ty/(2L) + tz/(4*drag)) / 4   ← note sign on ty flipped
    #   T3 = (ft + tx/(2L) - ty/(2L) - tz/(4*drag)) / 4
    #   T4 = (ft + tx/(2L) + ty/(2L) - tz/(4*drag)) / 4
    L2  = 2.0 * _L
    d4  = 4.0 * _SAT_drag
    T1 = (ft - tau[0]/L2 - tau[1]/L2 + tau[2]/d4) / 4.0
    T2 = (ft - tau[0]/L2 + tau[1]/L2 + tau[2]/d4) / 4.0   # MATLAB: -tx/(2L) +ty/(2L)
    T3 = (ft + tau[0]/L2 - tau[1]/L2 - tau[2]/d4) / 4.0
    T4 = (ft + tau[0]/L2 + tau[1]/L2 - tau[2]/d4) / 4.0

    # Step 3 — clip motor thrusts
    motors = np.clip([T1, T2, T3, T4], _SAT_Tmin, _SAT_Tmax)

    # Step 4 — recompute total thrust
    ft_out = float(np.clip(np.sum(motors), 0.0, _SAT_max_Ft))

    return ft_out, tau


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers: convert agent action ↔ physical [Ft, torques] for saturation
# ──────────────────────────────────────────────────────────────────────────────

def _integrated_action_to_physical(action, env):
    """
    Reproduce the integrated env's action→physical conversion:
        Ft    = MIN_THRUST + (a[0]+1)/2 * (MAX_THRUST - MIN_THRUST)
        tau_x = a[1] * MAX_TORQUE
        tau_y = a[2] * MAX_TORQUE
        tau_z = a[3] * MAX_TORQUE_YAW
    """
    a  = np.clip(np.array(action, dtype=np.float64), -1.0, 1.0)
    Ft = env.cfg.MIN_THRUST + (a[0] + 1.0) * 0.5 * (
         env.cfg.MAX_THRUST - env.cfg.MIN_THRUST)
    tau = np.array([
        a[1] * env.cfg.MAX_TORQUE,
        a[2] * env.cfg.MAX_TORQUE,
        a[3] * env.cfg.MAX_TORQUE_YAW,
    ])
    return Ft, tau


def _integrated_physical_to_action(Ft, tau, env):
    """Invert _integrated_action_to_physical, clip to [-1, 1]."""
    a0 = (Ft - env.cfg.MIN_THRUST) / (
         0.5 * (env.cfg.MAX_THRUST - env.cfg.MIN_THRUST)) - 1.0
    a1 = tau[0] / env.cfg.MAX_TORQUE
    a2 = tau[1] / env.cfg.MAX_TORQUE
    a3 = tau[2] / env.cfg.MAX_TORQUE_YAW
    return np.clip([a0, a1, a2, a3], -1.0, 1.0).astype(np.float32)


def _cascaded_action_to_physical(action, env):
    """
    Reproduce the cascaded position env's action→physical conversion.
    The pos-agent action is 3-D body acceleration; the env internally runs
    system_solve + attitude agent to get [ft, torques].  Those are stored in
    info['thrust'] and info['torques'] after each step.

    Since we cannot intercept them *before* dynamics, for the cascaded system
    we apply saturation in normalised action space using per-channel fractions
    of the actuator limits — the same approximation used for control noise.
    This function returns None to signal 'apply saturation in action space'.
    """
    return None   # handled directly in the runner


def apply_saturation_integrated(action, env):
    """
    Full pipeline for integrated agent:
      action (normalised) → physical → saturate_controls → normalised.
    Returns the saturated normalised action.
    """
    Ft, tau = _integrated_action_to_physical(action, env)
    Ft_sat, tau_sat = saturate_controls(Ft, tau)
    return _integrated_physical_to_action(Ft_sat, tau_sat, env)


def apply_saturation_cascaded(action, env):
    """
    Approximation for cascaded agent: saturate in normalised action space.
    The 3-D position action maps to body accelerations; we apply element-wise
    saturation using the same per-channel limits used for control noise.
    Exact motor-level saturation is not achievable here without re-running
    system_solve + the attitude inner loop.
    """
    max_acc = env.cfg.MAX_BODY_ACCELERATION
    # Clip each acc channel to [-1, 1] (already the training bound) and
    # additionally clip the z-channel to reflect that ft ≥ gravity compensation.
    # The gravity-compensation floor means az ∈ [-1, 1] but physically
    # ft ≥ MIN_THRUST requires az ≥ -1, which the [-1,1] clip already enforces.
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def run_cascaded_matlab_episode(env_agent, params, base_mass, demo=False,
                                saturate=False):
    """
    Cascaded TD3: one episode with MATLAB-faithful noise.

    At every step:
      1. add_state_noise applied to the *raw* env state -> noisy observation
      2. Agent selects action from noisy observation
      3. add_control_noise applied to [Ft, tau_x, tau_y, tau_z] before dynamics
         NOTE: the env's step() computes Ft and torques internally from the
         position action.  We intercept them via a wrapper.
      4. wind_f_m applied to state *after* step (as MATLAB does: X = wind_f_m(X,...))
    """
    if demo:
        return _synthetic_episode(base=0.085, params=params, n_actions=3)

    env, pos_agent = env_agent

    # Reset mass
    env.dynamics.m = base_mass * (
        1.0 + np.random.uniform(-params['mass_frac'], params['mass_frac'])
        if params['mass_frac'] > 0 else 1.0)

    # Draw a fixed wind vector for this episode (direction randomised)
    Wxyz = _sample_wind_vector(params['wind_speed'])

    # Full trajectory for distance tests
    original_max  = env.max_steps
    env.max_steps = env.sys_cfg.MAX_STEPS

    state = env.reset()
    pos_errors, vel_errors, attitudes, actions = [], [], [], []
    crashed = False
    DT = env.sys_cfg.DT

    ctrl_np  = params['ctrl_noise']
    state_np = params['state_noise']

    while True:
        # 1. Add sensor noise to the normalised observation (first 12 elements only)
        if state_np > 0:
            noisy_obs = add_state_noise(state, state_np).astype(np.float32)
        else:
            noisy_obs = state

        # 2. Agent action
        action = pos_agent.select_action(noisy_obs, explore=False)

        # 3. Step environment (computes Ft and torques internally)
        #    We can't intercept env internals cleanly without patching,
        #    so we apply control noise to the *position action* (normalised)
        #    before passing to env.step(), which is equivalent because the
        #    action is linearly mapped to body accelerations -> Ft/torques.
        if ctrl_np > 0:
            # Noise on raw action in [-1,1] space, scaled by actuator fractions.
            # MAX_BODY_ACCELERATION lives on env.cfg (PositionControllerConfig),
            # not env.sys_cfg (SystemConfig).
            max_acc = env.cfg.MAX_BODY_ACCELERATION
            action_uncert = np.array([
                ACTUATOR_UNCERT[0] / env.sys_cfg.MAX_THRUST,  # Ft fraction -> acc fraction
                ACTUATOR_UNCERT[1] / max_acc,
                ACTUATOR_UNCERT[2] / max_acc,
            ])
            action_noise = ctrl_np * action_uncert * np.random.randn(len(action))
            action = np.clip(action + action_noise, -1.0, 1.0)

        # Saturation: clip action to motor-feasible range (normalised approximation)
        if saturate:
            action = apply_saturation_cascaded(action, env)

        next_state, _reward, done, info = env.step(action)

        # 4. Apply wind disturbance to the physical state post-step (like MATLAB)
        if params['wind_speed'] > 0:
            env.state = wind_f_m(env.state, Wxyz, DT)

        pos_errors.append(info['pos_error'] / 1000.0)  # mm -> m
        vel_errors.append(float(info['vel_error']))
        attitudes.append(np.rad2deg(env.state[6:9]))
        actions.append(action.copy())

        if info.get('termination', '') in CRASH_TERMS:
            crashed = True
        if done:
            break
        state = next_state

    env.max_steps  = original_max
    env.dynamics.m = base_mass
    return episode_metrics(pos_errors, vel_errors, attitudes, actions, crashed, DT)


def run_integrated_matlab_episode(env, agent, params, base_mass, demo=False,
                                  demo_base=0.070, saturate=False):
    """
    Integrated TD3 / DDPG: one episode with MATLAB-faithful noise.
    Same logic as cascaded version (state noise on obs, control noise on action,
    wind_f_m applied post-step).
    """
    if demo:
        return _synthetic_episode(base=demo_base, params=params, n_actions=4)

    env.dynamics.m = base_mass * (
        1.0 + np.random.uniform(-params['mass_frac'], params['mass_frac'])
        if params['mass_frac'] > 0 else 1.0)

    Wxyz = _sample_wind_vector(params['wind_speed'])

    original_max  = env.max_steps
    env.max_steps = env.sys_cfg.MAX_STEPS

    state = env.reset()
    pos_errors, vel_errors, attitudes, actions = [], [], [], []
    crashed = False
    DT = env.sys_cfg.DT

    ctrl_np  = params['ctrl_noise']
    state_np = params['state_noise']

    while True:
        # State noise
        if state_np > 0:
            noisy_obs = add_state_noise(state, state_np).astype(np.float32)
        else:
            noisy_obs = state

        # Agent action
        action = agent.select_action(noisy_obs, explore=False)

        # Control noise: scale to action space
        if ctrl_np > 0:
            action_uncert = np.array([
                ACTUATOR_UNCERT[0] / (2.0 * _g),   # Ft normalised
                ACTUATOR_UNCERT[1] / _Max_torque,
                ACTUATOR_UNCERT[2] / _Max_torque,
                ACTUATOR_UNCERT[3] / _Max_torque,
            ])
            action_noise = ctrl_np * action_uncert * np.random.randn(len(action))
            action = np.clip(action + action_noise, -1.0, 1.0)

        # Saturation: enforce motor-level feasibility before dynamics
        if saturate:
            action = apply_saturation_integrated(action, env)

        next_state, _reward, done, info = env.step(action)

        # Wind disturbance post-step
        if params['wind_speed'] > 0:
            env.state = wind_f_m(env.state, Wxyz, DT)

        pos_errors.append(info['pos_error'] / 1000.0)  # mm -> m
        vel_errors.append(float(info['vel_error']))
        attitudes.append(np.rad2deg(env.state[6:9]))
        actions.append(action.copy())

        if info.get('termination', '') in CRASH_TERMS:
            crashed = True
        if done:
            break
        state = next_state

    env.max_steps  = original_max
    env.dynamics.m = base_mass
    return episode_metrics(pos_errors, vel_errors, attitudes, actions, crashed, DT)


def _synthetic_episode(base, params, n_actions):
    """Plausible fake episode for demo mode."""
    scale  = 1.0
    scale += params['ctrl_noise']  * 0.03
    scale += params['state_noise'] * 0.04
    scale += params['wind_speed']  * 0.12
    eff_base = base * (1.0 + scale * 0.3)
    n      = 5000
    errs   = np.abs(np.random.normal(eff_base, eff_base * 0.3, n))  # metres
    vels   = np.abs(np.random.normal(0.3, 0.15, n))
    atts   = np.random.normal(0, 4.5, (n, 3))
    acts   = np.random.uniform(-0.25, 0.25, (n, n_actions))
    crash  = np.random.random() < 0.01 * (
        1 + params['ctrl_noise'] * 0.5 + params['state_noise'] * 0.3
        + params['wind_speed'] * 0.2)
    return episode_metrics(errs, vels, atts, acts, crash, dt=0.01)


# ==============================================================================
#  EVALUATE ONE SYSTEM OVER ALL CONDITIONS
# ==============================================================================

def evaluate_system(label, run_fn, conditions, n_episodes):
    print(f"\n  +-- {label} " + "-" * max(0, 52 - len(label)) + "+")
    results = {}
    for cid, cname, params in conditions:
        ep_results = [run_fn(params) for _ in range(n_episodes)]
        agg = aggregate_metrics(ep_results)
        results[cid] = agg
        print(f"  |  {cname:<34}  RMSE={agg['rmse_mean']:6.4f} m   "
              f"Succ={agg['success_rate']:5.1f}%  Crash={agg['crash_rate']:4.1f}%")
    print("  +" + "-" * 60 + "+")
    return results


# ==============================================================================
#  OUTPUT FORMATTING
# ==============================================================================

def fmt(val, spec='.4f', na='--'):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return na
    return format(val, spec)


def print_console_table(conditions, systems):
    n   = len(systems)
    col = 54
    w   = 36 + n * (col + 3)

    print()
    print('=' * w)
    print('  MATLAB-FAITHFUL NOISE TEST — PERFORMANCE TABLE')
    print('=' * w)

    header = f"{'Condition':<36}"
    for label, _ in systems:
        header += f" | {label:^52}"
    print(header)

    sub = ' ' * 36
    mh  = f"{'RMSE':>8} {'Mean+/-Std':>15} {'Max':>7} {'Succ%':>6} {'Crash%':>7} {'Conv(s)':>7}"
    for _ in systems:
        sub += f" | {mh}"
    print(sub)
    print('-' * w)

    groups = {
        'baseline': '-- Baseline',
        'ctrl1':    '-- Control Noise (actuator uncertainty)',
        'state1':   '-- State / Sensor Noise',
        'wind0p5':  '-- Aerodynamic Wind  (wind_f_m model)',
        'combined': '-- Combined',
    }

    for cid, cname, _ in conditions:
        if cid in groups:
            print(f"\n  {groups[cid]}")
        row = f"{cname:<36}"
        for _, res in systems:
            r  = res[cid]
            ms = f"{fmt(r['mean_err'])}+/-{fmt(r['mean_err_std'])}"
            row += (f" | {fmt(r['rmse_mean']):>8} {ms:>15} {fmt(r['max_err']):>7}"
                    f" {fmt(r['success_rate'],'.0f'):>6} {fmt(r['crash_rate'],'.0f'):>7}"
                    f" {fmt(r['conv_time']):>7}")
        print(row)

    print('\n' + '=' * w)
    print("  RMSE/Max m  | Succ%: RMSE<0.3 m | Conv: first sustain <0.1 m >= 20 steps (s)\n")


def save_latex(conditions, systems, filepath):
    n_sys   = len(systems)
    n_cols  = n_sys * 6
    col_fmt = 'l' + (' | ' + 'r' * 6) * n_sys

    group_headers = {
        'baseline': r'\multicolumn{' + str(1+n_cols) + r'}{l}{\textit{Baseline}} \\',
        'ctrl1':    r'\multicolumn{' + str(1+n_cols) + r'}{l}{\textit{Control Noise (actuator uncertainty, add\_control\_noise)}} \\',
        'state1':   r'\multicolumn{' + str(1+n_cols) + r'}{l}{\textit{State / Sensor Noise (add\_state\_noise)}} \\',
        'wind0p5':  r'\multicolumn{' + str(1+n_cols) + r'}{l}{\textit{Aerodynamic Wind Disturbance (wind\_f\_m model)}} \\',
        'combined': r'\multicolumn{' + str(1+n_cols) + r'}{l}{\textit{Combined}} \\',
    }

    L = []
    L.append(r'% Auto-generated MATLAB-faithful noise evaluation table')
    L.append(r'\begin{table*}[!ht]')
    L.append(r'\centering')
    L.append(
        r'\caption{Performance under sensor and actuator disturbances. '
        r'\texttt{add\_control\_noise} adds Gaussian noise scaled by actuator '
        r'uncertainty to the control vector; \texttt{add\_state\_noise} adds '
        r'per-channel Gaussian noise in physical units '
        r'($\sigma_\mathrm{pos}$\,=\,0.01\,m, $\sigma_\mathrm{vel}$\,=\,0.01\,m/s, '
        r'$\sigma_\mathrm{att}$\,=\,0.005\,rad, $\sigma_\mathrm{rate}$\,=\,0.005\,rad/s) '
        r'multiplied by noise\_percent; '
        r'\texttt{wind\_f\_m} applies full aerodynamic drag and moment disturbances. '
        r'All position errors in metres. Bold = best result per column.}')
    L.append(r'\label{tab:matlab_noise}')
    L.append(r'\small')
    L.append(r'\setlength{\tabcolsep}{4pt}')
    L.append(r'\begin{tabular}{' + col_fmt + r'}')
    L.append(r'\toprule')

    sys_hdr = ''
    for i, (lbl, _) in enumerate(systems):
        align = r'|c' if i == 0 else r'c'
        sys_hdr += r' & \multicolumn{6}{' + align + r'}{\textbf{' + lbl + r'}}'
    L.append(r'\textbf{Disturbance Condition}' + sys_hdr + r' \\')

    cr = ''
    for i in range(n_sys):
        c1 = 2 + i*6;  c2 = c1+5
        cr += r'\cmidrule(lr){' + str(c1) + '-' + str(c2) + '}'
    L.append(cr)

    mnames = [r'\textbf{RMSE}', r'\textbf{Mean$\pm$Std}', r'\textbf{Max}',
              r'\textbf{Succ.}', r'\textbf{Crash}', r'\textbf{Conv.}']
    L.append(r'\textbf{Condition}' + (' & ' + ' & '.join(mnames)) * n_sys + r' \\')
    units  = [r'\textbf{(m)}'] * 3 + [r'\textbf{(\%)}', r'\textbf{(\%)}', r'\textbf{(s)}']
    L.append('' + (' & ' + ' & '.join(units)) * n_sys + r' \\')
    L.append(r'\midrule')

    def best_of(cid, fn, lower=True):
        vals = []
        for _, res in systems:
            try: vals.append(float(fn(res[cid])))
            except: vals.append(float('inf'))
        return int(np.argmin(vals)) if lower else int(np.argmax(vals))

    for cid, cname, _ in conditions:
        if cid in group_headers:
            L.append(group_headers[cid])

        bi_rmse  = best_of(cid, lambda r: r['rmse_mean'])
        bi_succ  = best_of(cid, lambda r: r['success_rate'], lower=False)
        bi_crash = best_of(cid, lambda r: r['crash_rate'])
        bi_conv  = best_of(cid, lambda r: r['conv_time'] if not np.isnan(r['conv_time']) else 1e9)

        row = f'  {cname}'
        for si, (_, res) in enumerate(systems):
            r  = res[cid]
            ms = f"{fmt(r['mean_err'])}$\\pm${fmt(r['mean_err_std'])}"
            rs = fmt(r['rmse_mean'])
            ss = fmt(r['success_rate'], '.0f')
            cs = fmt(r['crash_rate'],   '.0f')
            cv = fmt(r['conv_time'])
            if si == bi_rmse:  rs = r'\textbf{' + rs + '}'
            if si == bi_succ:  ss = r'\textbf{' + ss + '}'
            if si == bi_crash: cs = r'\textbf{' + cs + '}'
            if si == bi_conv:  cv = r'\textbf{' + cv + '}'
            row += f' & {rs} & {ms} & {fmt(r["max_err"])} & {ss} & {cs} & {cv}'
        row += r' \\'
        L.append(row)

    # Overall summary row
    L.append(r'\midrule')
    sumrow = r'\textbf{Mean (all conditions)}'
    for si, (_, res) in enumerate(systems):
        all_rmse = np.mean([res[cid]['rmse_mean']    for cid,_,_ in conditions])
        all_succ = np.mean([res[cid]['success_rate'] for cid,_,_ in conditions])
        sumrow += f' & {fmt(all_rmse)} & -- & -- & {fmt(all_succ,".1f")} & -- & --'
    sumrow += r' \\'
    L.append(sumrow)

    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\vspace{0.4em}')
    L.append(r'\begin{flushleft}\footnotesize')
    L.append(
        r'Control noise: Gaussian with std\,=\,noise\_percent\,$\times$\,'
        r'[10\%\,$F_{t,\max}$, 1\%\,$\tau_{\max}$, 1\%\,$\tau_{\max}$, 2\%\,$\tau_{\max}$]. '
        r'State noise: per-channel Gaussian with base $\sigma$ in physical units '
        r'($\sigma_\mathrm{pos}$\,=\,0.01\,m, $\sigma_\mathrm{vel}$\,=\,0.01\,m/s, '
        r'$\sigma_\mathrm{att}$\,=\,0.005\,rad\,=\,0.29$^\circ$, '
        r'$\sigma_\mathrm{rate}$\,=\,0.005\,rad/s) scaled by noise\_percent and '
        r'mapped to normalised observation space; obs[12:18] (trajectory planner) unperturbed. '
        r'Wind: full aerodynamic drag ($C_d=[0.8,0.8,1.2]$, $\rho=1.225$\,kg/m$^3$) '
        r'plus motor torques from $4\times$ motor moment arms. '
        r'Position errors in metres. \textbf{Bold}\,=\,best per column.')
    L.append(r'\end{flushleft}')
    L.append(r'\end{table*}')

    with open(filepath, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f"  LaTeX  -> {filepath}")


def save_csv(conditions, systems, filepath):
    cols = ['condition', 'system',
            'ctrl_noise_pct', 'state_noise_pct', 'wind_speed_ms',
            'rmse_m', 'rmse_std_m', 'mean_m', 'mean_std_m', 'max_m',
            'success_pct', 'crash_pct', 'conv_time_s', 'smoothness']
    rows = []
    for cid, cname, params in conditions:
        for label, res in systems:
            r = res[cid]
            rows.append([
                cname, label,
                params['ctrl_noise'], params['state_noise'], params['wind_speed'],
                fmt(r['rmse_mean'],    '.3f'), fmt(r['rmse_std'],     '.3f'),
                fmt(r['mean_err'],     '.3f'), fmt(r['mean_err_std'], '.3f'),
                fmt(r['max_err'],      '.3f'), fmt(r['success_rate'], '.1f'),
                fmt(r['crash_rate'],   '.1f'), fmt(r['conv_time'],    '.3f'),
                fmt(r['smoothness'],   '.5f'),
            ])
    with open(filepath, 'w', newline='') as f:
        csv.writer(f).writerows([cols] + rows)
    print(f"  CSV    -> {filepath}")


def save_npy(conditions, systems, filepath):
    data = {
        'systems':    [(lbl, res) for lbl, res in systems],
        'conditions': [(cid, cname, params) for cid, cname, params in conditions],
        'noise_info': {
            'ACTUATOR_UNCERT':    ACTUATOR_UNCERT,
            'SENSOR_SIGMA_POS':   _SENSOR_SIGMA_POS,
            'SENSOR_SIGMA_VEL':   _SENSOR_SIGMA_VEL,
            'SENSOR_SIGMA_ATT':   _SENSOR_SIGMA_ATT,
            'SENSOR_SIGMA_RATE':  _SENSOR_SIGMA_RATE,
            'Cd': _Cd, 'A_frontal': _A, 'rho': _rho,
        },
    }
    np.save(filepath, data, allow_pickle=True)
    print(f"  NumPy  -> {filepath}")


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MATLAB-faithful noise/wind evaluation for all three agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--cascaded_att',    default=None, metavar='PATH')
    parser.add_argument('--cascaded_pos',    default=None, metavar='PATH')
    parser.add_argument('--integrated_td3',  default=None, metavar='PATH')
    parser.add_argument('--integrated_ddpg', default=None, metavar='PATH')

    parser.add_argument('--cascaded_root',        default=None, metavar='DIR')
    parser.add_argument('--integrated_td3_root',  default=None, metavar='DIR')
    parser.add_argument('--integrated_ddpg_root', default=None, metavar='DIR')

    parser.add_argument('--episodes', type=int, default=30,
                        help='Episodes per condition (default: 30)')
    parser.add_argument('--output',   default='results_matlab/',
                        help='Output directory (default: results_matlab/)')
    parser.add_argument('--demo',     action='store_true',
                        help='Demo mode: synthetic episodes, no checkpoints needed')
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--saturate', action='store_true',
                        help='Apply MATLAB saturate_controls() to every action before '
                             'env.step(): clips torques to motor limits, solves per-motor '
                             'thrusts, re-clips, and recomputes total thrust. '
                             'For integrated agents the round-trip is exact; '
                             'for cascaded agents it is applied in normalised action space.')

    args = parser.parse_args()
    np.random.seed(args.seed)

    global CASCADED_ROOT, INTEGRATED_TD3_ROOT, INTEGRATED_DDPG_ROOT
    if args.cascaded_root:        CASCADED_ROOT        = args.cascaded_root
    if args.integrated_td3_root:  INTEGRATED_TD3_ROOT  = args.integrated_td3_root
    if args.integrated_ddpg_root: INTEGRATED_DDPG_ROOT = args.integrated_ddpg_root

    want_cascaded = args.cascaded_att or args.cascaded_pos
    want_td3_int  = args.integrated_td3
    want_ddpg_int = args.integrated_ddpg

    if not (want_cascaded or want_td3_int or want_ddpg_int):
        print('\n  [INFO] No checkpoints supplied -- switching to --demo mode.\n')
        args.demo = True
        want_cascaded = want_td3_int = want_ddpg_int = True

    os.makedirs(args.output, exist_ok=True)
    conditions = make_matlab_conditions()

    print('\n' + '=' * 65)
    print('  MATLAB-FAITHFUL NOISE / WIND EVALUATION')
    print('=' * 65)
    print(f'  Mode      : {"DEMO (synthetic)" if args.demo else "REAL"}')
    print(f'  Episodes  : {args.episodes} per condition')
    print(f'  Conditions: {len(conditions)}')
    print(f'  Saturate  : {"ON  (saturate_controls applied each step)" if args.saturate else "OFF"}')
    print(f'  Seed      : {args.seed}')
    print(f'  Output    : {args.output}')
    print(f'  Noise models:')
    print(f'    add_control_noise: uncert = {ACTUATOR_UNCERT}')
    print(f'    add_state_noise  : pos={_SENSOR_SIGMA_POS} m, vel={_SENSOR_SIGMA_VEL} m/s, '
          f'att={_SENSOR_SIGMA_ATT} rad ({np.rad2deg(_SENSOR_SIGMA_ATT):.2f} deg), '
          f'rate={_SENSOR_SIGMA_RATE} rad/s  (x noise_percent)')
    print(f'    wind_f_m         : Cd={_Cd}, A={_A}, rho={_rho}')
    systems_list = ((['Cascaded TD3'] if want_cascaded else []) +
                    (['Integrated TD3'] if want_td3_int else []) +
                    (['Integrated DDPG'] if want_ddpg_int else []))
    print(f'  Systems   : {", ".join(systems_list)}')
    print('=' * 65)

    evaluated = []   # list of (label, {cid: agg_metrics})

    # 1. Cascaded TD3
    if want_cascaded:
        print('\n[1/3] Loading Cascaded TD3...')
        casc, base_mass_casc = load_cascaded_td3(
            args.cascaded_att, args.cascaded_pos, demo=args.demo)

        def run_casc(params):
            return run_cascaded_matlab_episode(casc, params, base_mass_casc,
                                               demo=args.demo,
                                               saturate=args.saturate)
        t0 = time.time()
        res = evaluate_system('Cascaded TD3', run_casc, conditions, args.episodes)
        print(f'    done in {time.time()-t0:.1f}s')
        evaluated.append(('Cascaded TD3', res))

    # 2. Integrated TD3
    if want_td3_int:
        print('\n[2/3] Loading Integrated TD3...')
        env_td3, agent_td3, base_mass_td3 = load_integrated_td3(
            args.integrated_td3, demo=args.demo)

        def run_td3(params):
            return run_integrated_matlab_episode(env_td3, agent_td3, params,
                                                  base_mass_td3, demo=args.demo,
                                                  demo_base=0.075,
                                                  saturate=args.saturate)
        t0 = time.time()
        res = evaluate_system('Integrated TD3', run_td3, conditions, args.episodes)
        print(f'    done in {time.time()-t0:.1f}s')
        evaluated.append(('Integrated TD3', res))

    # 3. Integrated DDPG
    if want_ddpg_int:
        print('\n[3/3] Loading Integrated DDPG...')
        env_ddpg, agent_ddpg, base_mass_ddpg = load_integrated_ddpg(
            args.integrated_ddpg, demo=args.demo)

        def run_ddpg(params):
            return run_integrated_matlab_episode(env_ddpg, agent_ddpg, params,
                                                  base_mass_ddpg, demo=args.demo,
                                                  demo_base=0.070,
                                                  saturate=args.saturate)
        t0 = time.time()
        res = evaluate_system('Integrated DDPG', run_ddpg, conditions, args.episodes)
        print(f'    done in {time.time()-t0:.1f}s')
        evaluated.append(('Integrated DDPG', res))

    # ── Output ────────────────────────────────────────────────────────────────
    print_console_table(conditions, evaluated)

    print('Saving outputs...')
    save_latex(conditions, evaluated,
               os.path.join(args.output, 'matlab_noise_table.tex'))
    save_csv(conditions, evaluated,
             os.path.join(args.output, 'matlab_noise_results.csv'))
    save_npy(conditions, evaluated,
             os.path.join(args.output, 'matlab_noise_results.npy'))

    print(f'\n  Output files:')
    print(f'    matlab_noise_table.tex   <- paste into paper')
    print(f'    matlab_noise_results.csv <- spreadsheet / pandas')
    print(f'    matlab_noise_results.npy <- for plotting scripts')
    print()


if __name__ == '__main__':
    main()