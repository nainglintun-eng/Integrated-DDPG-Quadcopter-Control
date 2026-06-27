"""
Main Script – Integrated Single-Agent DDPG Quadcopter Control
==============================================================
One DDPG agent maps  observation (18-dim)  →  U = [Ft, tau_x, tau_y, tau_z].

Run:
    python main.py

What it does:
  1.  Prints system info (inertia, thrust / torque limits, trajectory)
  2.  Trains the integrated DDPG agent (curriculum, early stopping)
  3.  Saves the best weights as:
          results/<run>/integrated_controller_final.pth   (PyTorch)
          results/<run>/z_DDPG_Integrated.mat             (MATLAB / Simulink)
  4.  Runs a final evaluation and prints metrics
  5.  Plots training curves and 3-D trajectory

MATLAB .mat format (load with  S = load('z_DDPG_Integrated.mat')):
    agent.actor_W1, agent.actor_b1, …
    agent.critic_W1, …         <- single Q-network (DDPG, not twin)
    agent.max_action, agent.state_dim, agent.action_dim

Resume from checkpoint:
    RESUME_FROM=results/run_XYZ/integrated_phase_2.pth python main.py
"""

import numpy as np
import torch
import random
import os
import sys
import scipy.io
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config               import (SystemConfig, IntegratedAgentConfig,
                                          TrainingConfig, get_trajectory_function)
from agents.ddpg_agent            import DDPGAgent
from environments.integrated_env  import IntegratedEnv
from training.trainer             import train_integrated_controller
from utils.visualization          import plot_training_curves, plot_trajectory_3d
from utils.evaluation             import evaluate_system


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def print_system_info():
    cfg     = SystemConfig()
    int_cfg = IntegratedAgentConfig()

    print('\n' + '='*80)
    print('INTEGRATED SINGLE-AGENT DDPG – QUADCOPTER TRAJECTORY TRACKING')
    print('='*80)
    print('\nMISSION PROFILE:')
    print(f'  Trajectory    : Expanding Helix')
    print(f'  R(t)          = {cfg.A} * (1 - exp(-{cfg.B}*t))')
    print(f'  x(t)          = R(t) * cos({cfg.OMEGA}*t)')
    print(f'  y(t)          = R(t) * sin({cfg.OMEGA}*t)')
    print(f'  z(t)          = {cfg.VZ}*t')
    print(f'  Duration      : {cfg.MAX_STEPS * cfg.DT:.0f} s')
    print(f'  Control rate  : {1/cfg.DT:.0f} Hz  (dt = {cfg.DT} s)')
    print(f'  Max steps     : {cfg.MAX_STEPS:,}')

    print(f'\nQUADCOPTER SPECS:')
    print(f'  Mass          : {cfg.MASS} kg')
    print(f'  Ixx/Iyy/Izz   : {cfg.IXX}/{cfg.IYY}/{cfg.IZZ} kg·m²')
    print(f'  L (arm)       : {cfg.L} m')
    print(f'  MAX_THRUST    : {cfg.MAX_THRUST:.4f} N  (= 2·m·g)')
    print(f'  MIN_THRUST    : {cfg.MIN_THRUST:.4f} N  (= 0.1·m·g)')
    print(f'  MAX_TORQUE    : {cfg.MAX_TORQUE:.4f} Nm')
    print(f'  MAX_TORQUE_YAW: {cfg.MAX_TORQUE_YAW:.4f} Nm')

    print(f'\nAGENT (INTEGRATED DDPG):')
    print(f'  Architecture  : Residual Actor + Single-Q Critic')
    print(f'                  (3 res blocks, hidden={int_cfg.HIDDEN_DIMS})')
    print(f'  Observation   : {int_cfg.STATE_DIM}-dim')
    print(f'  Action        : {int_cfg.ACTION_DIM}-dim  [Ft, tau_x, tau_y, tau_z]')
    print(f'  Actor scale   : ±{int_cfg.ACTION_SCALE:.4f}  (per-channel clamping in env)')
    print(f'  Exploration   : Ornstein-Uhlenbeck noise')
    print(f'    theta        : {int_cfg.OU_THETA}')
    print(f'    sigma (init) : {int_cfg.OU_SIGMA}  →  min {int_cfg.OU_SIGMA_MIN}')
    print(f'    dt           : {int_cfg.OU_DT} s')

    print(f'\nTRAINING:')
    print(f'  Wind training : {"ENABLED" if cfg.WIND_TRAINING else "DISABLED"}')
    print(f'  Random start  : {"ENABLED" if cfg.RANDOM_START  else "DISABLED"}')
    print(f'  Phases        : {len(int_cfg.CURRICULUM_PHASES)}')
    total_ep = sum(p["episodes"] for p in int_cfg.CURRICULUM_PHASES)
    print(f'  Total episodes: {total_ep}')
    print('='*80)


def save_agent_as_mat(agent, filepath, agent_key='agent'):
    """
    Export all Linear-layer weights to a .mat file for MATLAB / Simulink.

    DDPG has a single Q-network, so the keys are:
        actor_W1, actor_b1, actor_W2, …
        critic_W1, critic_b1, …        (single Q, not twin)

    MATLAB:
        S  = load('z_DDPG_Integrated.mat');
        ag = S.agent;
        W1 = ag.actor_W1;   b1 = ag.actor_b1;
        …
    """
    def _extract(model):
        out, idx = {}, 1
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                out[f'W{idx}'] = m.weight.detach().cpu().numpy()
                out[f'b{idx}'] = m.bias.detach().cpu().numpy()
                idx += 1
        return out

    mat_dict = {}
    for k, v in _extract(agent.actor).items():
        mat_dict[f'actor_{k}'] = v
    # Single Q-network  (agent.critic.net)
    for k, v in _extract(agent.critic).items():
        mat_dict[f'critic_{k}'] = v

    mat_dict['max_action'] = np.array([agent.max_action])
    mat_dict['state_dim']  = np.array([agent.state_dim])
    mat_dict['action_dim'] = np.array([agent.action_dim])

    scipy.io.savemat(filepath, {agent_key: mat_dict})
    print(f'  → .mat saved: {filepath}  ({len(mat_dict)} arrays, key="{agent_key}")')


def main():
    print_system_info()

    set_seeds(SystemConfig.RANDOM_SEED)
    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir   = f'results/run_{timestamp}'
    os.makedirs(run_dir,            exist_ok=True)
    os.makedirs(f'{run_dir}/plots', exist_ok=True)
    print(f'\nResults → {run_dir}')

    # ── Environment ──
    int_cfg = IntegratedAgentConfig()
    env     = IntegratedEnv(config=int_cfg)

    # ── DDPG Agent ──
    agent = DDPGAgent(
        state_dim   = int_cfg.STATE_DIM,
        action_dim  = int_cfg.ACTION_DIM,
        max_action  = int_cfg.ACTION_SCALE,
        hidden_dims = int_cfg.HIDDEN_DIMS,
        actor_lr    = int_cfg.ACTOR_LR,
        critic_lr   = int_cfg.CRITIC_LR,
        gamma       = int_cfg.GAMMA,
        tau         = int_cfg.TAU,
        ou_theta    = int_cfg.OU_THETA,
        ou_sigma    = int_cfg.OU_SIGMA,
        ou_dt       = int_cfg.OU_DT,
        buffer_size = int_cfg.BUFFER_SIZE,
    )
    agent.warmup_steps = int_cfg.WARMUP_STEPS
    agent.batch_size   = int_cfg.BATCH_SIZE

    # ── Resume ──
    resume_path = os.environ.get('RESUME_FROM', '')
    if resume_path and os.path.isfile(resume_path):
        agent.load(resume_path)
        print(f'\n✓ Resumed from: {resume_path}')

    # ── Training ──
    print('\n' + '='*80)
    print('TRAINING  –  DDPG')
    print('='*80)

    stats = train_integrated_controller(
        env=env, agent=agent, config=int_cfg, save_dir=run_dir)

    # ── Save weights ──
    final_pth = f'{run_dir}/integrated_controller_final.pth'
    agent.save(final_pth)
    print(f'\n✓ PyTorch weights  → {final_pth}')

    final_mat = f'{run_dir}/z_DDPG_Integrated.mat'
    save_agent_as_mat(agent, final_mat, agent_key='agent')
    print(f'✓ MATLAB .mat      → {final_mat}')

    plot_training_curves(stats,
                         f'{run_dir}/plots/integrated_training.png',
                         'Integrated DDPG Training')

    # ── Evaluation ──
    print('\n' + '='*80)
    print('FINAL EVALUATION')
    print('='*80)

    eval_env = IntegratedEnv(config=int_cfg)
    eval_env.set_trajectory_scale(1.0)
    eval_env.set_wind([0.5, 0.5, 0.3])

    eval_results = evaluate_system(
        env=eval_env, position_agent=agent, num_episodes=10, save_dir=run_dir)

    print('\n' + '='*80)
    print('TRAINING COMPLETE – FINAL METRICS')
    print('='*80)
    print(f'  Mean pos error : {eval_results["mean_pos_error"]:.1f} mm')
    print(f'  RMSE           : {eval_results["rmse"]:.1f} mm')
    print(f'  Max error      : {eval_results["max_error"]:.1f} mm')
    print(f'  Success rate   : {eval_results["success_rate"]:.1%}')
    print(f'  Max roll       : {eval_results["max_roll"]:.1f}°')
    print(f'  Max pitch      : {eval_results["max_pitch"]:.1f}°')

    mean_m = eval_results['mean_pos_error'] / 1000
    if   mean_m < 0.5: print('\n✓✓✓ EXCELLENT PERFORMANCE!')
    elif mean_m < 0.8: print('\n✓✓  GOOD PERFORMANCE!')
    elif mean_m < 1.2: print('\n✓   ACCEPTABLE PERFORMANCE')
    else:              print('\n⚠   NEEDS IMPROVEMENT')

    print(f'\nAll results → {run_dir}')
    print('='*80)

    plot_trajectory_3d(eval_env, f'{run_dir}/plots/final_trajectory.png')
    np.save(f'{run_dir}/evaluation_results.npy', eval_results)
    print('\n✓ Training pipeline complete!')


if __name__ == '__main__':
    main()
