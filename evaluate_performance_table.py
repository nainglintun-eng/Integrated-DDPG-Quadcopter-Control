"""
evaluate_performance_table.py
==============================
Evaluate any combination of:
  1. Cascaded TD3          (attitude agent + position agent)
  2. Integrated TD3        (single agent, 4-dim action)
  3. Integrated DDPG       (single agent, 4-dim action)

You can also evaluate just ONE system by only providing its checkpoints.

Disturbance conditions
-----------------------
  Baseline   : No disturbance
  Wind       : 1 % – 5 % (wind_std scales linearly with percentage)
  System     : Sensor noise | Mass uncertainty ±10% | Actuator noise 3%
  Combined   : Wind 3% + Sensor noise

Metrics per condition
----------------------
  RMSE (mm)         root-mean-square 3-D position error
  Mean ± Std (mm)   mean ± std of per-step error
  Max (mm)          mean peak per-episode error
  Succ. (%)         episodes where RMSE < 300 mm
  Crash (%)         episodes ending in crash / OOB / velocity limit
  Conv. (s)         mean time to first sustain < 100 mm for ≥ 20 steps

Outputs
--------
  results/performance_table.tex   ready-to-paste LaTeX table
  results/performance_results.csv flat CSV (Excel / pandas friendly)
  results/performance_results.npy numpy dict (for plotting)

Usage examples
---------------
  # All three systems:
  python evaluate_performance_table.py \\
      --cascaded_att  runs/attitude_controller_final.pth \\
      --cascaded_pos  runs/position_controller_final.pth \\
      --integrated_td3   runs/td3_integrated_final.pth \\
      --integrated_ddpg  runs/ddpg_integrated_final.pth \\
      --episodes 50

  # Cascaded only:
  python evaluate_performance_table.py \\
      --cascaded_att  runs/attitude_controller_final.pth \\
      --cascaded_pos  runs/position_controller_final.pth \\
      --episodes 50

  # Integrated DDPG only:
  python evaluate_performance_table.py \\
      --integrated_ddpg  runs/ddpg_integrated_final.pth \\
      --episodes 50

  # Demo (no checkpoints needed – uses random weights to test the pipeline):
  python evaluate_performance_table.py --demo --episodes 10
"""

import argparse
import os
import sys
import csv
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# ── repo roots ────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
CASCADED_ROOT   = os.path.join(SCRIPT_DIR, 'cascaded',      'agents_modified')
INTEGRATED_ROOT = os.path.join(SCRIPT_DIR, 'integrated_ddpg', 'integrated_agent')

# ══════════════════════════════════════════════════════════════════════════════
#  DISTURBANCE CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

def make_conditions():
    """Return list of (id, display_name, params)."""
    def wind(pct):
        s = pct / 100.0 * 0.1          # 1% → std = 0.001  (matches dynamics convention)
        return {
            'wind_std':       [s, s, s * 0.6],
            'wind_gust_prob': min(0.02 * pct, 0.08),
            'wind_gust_mag':  0.0002 * pct,
        }

    def base_noise():
        return {'sensor_noise_std': 0.0, 'mass_noise_frac': 0.0, 'actuator_noise': 0.0}

    def no_wind():
        return {'wind_std': [0, 0, 0], 'wind_gust_prob': 0.0, 'wind_gust_mag': 0.0}

    return [
        # ── Baseline ─────────────────────────────────────────────────────────
        ('noisefree',  'No Disturbance',
         {**no_wind(), **base_noise()}),

        # ── Wind ─────────────────────────────────────────────────────────────
        ('wind1',  'Wind 1%',  {**wind(1), **base_noise()}),
        ('wind2',  'Wind 2%',  {**wind(2), **base_noise()}),
        ('wind3',  'Wind 3%',  {**wind(3), **base_noise()}),
        ('wind4',  'Wind 4%',  {**wind(4), **base_noise()}),
        ('wind5',  'Wind 5%',  {**wind(5), **base_noise()}),

        # ── System noise ─────────────────────────────────────────────────────
        ('sensor',    'Sensor Noise',
         {**no_wind(), 'sensor_noise_std': 0.01,
          'mass_noise_frac': 0.0, 'actuator_noise': 0.0}),

        ('mass_unc',  'Mass Uncertainty ±10%',
         {**no_wind(), 'sensor_noise_std': 0.0,
          'mass_noise_frac': 0.10, 'actuator_noise': 0.0}),

        ('actuator',  'Actuator Noise 3%',
         {**no_wind(), 'sensor_noise_std': 0.0,
          'mass_noise_frac': 0.0, 'actuator_noise': 0.03}),

        # ── Combined ─────────────────────────────────────────────────────────
        ('combined',  'Wind 3% + Sensor Noise',
         {**wind(3), 'sensor_noise_std': 0.01,
          'mass_noise_frac': 0.0, 'actuator_noise': 0.0}),
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED: APPLY DISTURBANCE TO ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

def apply_disturbance(env, params, base_mass=1.0):
    """Configure wind and mass uncertainty on any env that has set_wind() and dynamics.m."""
    env.set_wind(params['wind_std'])
    env.dynamics.set_wind(
        enabled=any(s > 0 for s in params['wind_std']),
        std=params['wind_std'],
        mean=[0, 0, 0],
        gust_prob=params['wind_gust_prob'],
        gust_mag=params['wind_gust_mag'],
    )
    frac = params['mass_noise_frac']
    env.dynamics.m = base_mass * (1.0 + np.random.uniform(-frac, frac)) if frac > 0 else base_mass


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED: COMPUTE METRICS FROM ONE EPISODE
# ══════════════════════════════════════════════════════════════════════════════

CRASH_TERMS = ('crash', 'out_of_bounds', 'altitude_limit',
               'excessive_velocity', 'excessive_rate')

def episode_metrics(pos_errors, attitudes, actions, crashed, dt):
    """Aggregate per-step lists into summary metrics dict."""
    errs = np.array(pos_errors, dtype=float)
    atts = np.array(attitudes,  dtype=float)
    acts = np.array(actions,    dtype=float)

    rmse  = float(np.sqrt(np.mean(errs ** 2)))
    mean  = float(np.mean(errs))
    std   = float(np.std(errs))
    maxi  = float(np.max(errs))

    # First time < 100 mm sustained for ≥ 20 consecutive steps
    conv_t = float('nan')
    for i in range(len(errs) - 19):
        if np.all(errs[i:i + 20] < 100.0):
            conv_t = i * dt
            break
    if np.isnan(conv_t) and not crashed:
        conv_t = len(errs) * dt   # never converged but didn't crash

    smooth = float(np.mean(np.abs(np.diff(acts, axis=0)))) if len(acts) > 1 else 0.0

    return {
        'rmse':     rmse,
        'mean':     mean,
        'std':      std,
        'max':      maxi,
        'conv_t':   conv_t,
        'crashed':  crashed,
        'smooth':   smooth,
        'roll':     float(np.mean(np.abs(atts[:, 0]))) if len(atts) else 0.0,
        'pitch':    float(np.mean(np.abs(atts[:, 1]))) if len(atts) else 0.0,
    }


def aggregate_metrics(episode_list):
    """Aggregate a list of per-episode metric dicts into condition-level stats."""
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
        'success_rate': float(np.mean([1 if r < 300 else 0 for r in rmses])) * 100,
        'crash_rate':   crashes / n * 100,
        'conv_time':    float(np.mean(convs)) if convs else float('nan'),
        'smoothness':   float(np.mean([m['smooth'] for m in episode_list])),
        'mean_roll':    float(np.mean([m['roll']   for m in episode_list])),
        'mean_pitch':   float(np.mean([m['pitch']  for m in episode_list])),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_cascaded_td3(att_ckpt, pos_ckpt, demo=False):
    """
    Load the cascaded TD3 system.
    Returns (position_env_with_attitude_agent, base_mass).
    """
    if demo:
        return None, 1.0

    if CASCADED_ROOT not in sys.path:
        sys.path.insert(0, CASCADED_ROOT)

    from agents.td3_agent import TD3Agent
    from environments.position_env import PositionEnv
    from configs.config import (SystemConfig, AttitudeControllerConfig,
                                PositionControllerConfig)

    att_cfg = AttitudeControllerConfig()
    pos_cfg = PositionControllerConfig()

    att_agent = TD3Agent(
        state_dim   = att_cfg.STATE_DIM,
        action_dim  = att_cfg.ACTION_DIM,
        max_action  = 1.0,
        hidden_dims = [att_cfg.HIDDEN_DIMS, att_cfg.HIDDEN_DIMS],
        actor_lr    = att_cfg.ACTOR_LR,
        critic_lr   = att_cfg.CRITIC_LR,
        gamma       = att_cfg.GAMMA,
        tau         = att_cfg.TAU,
        policy_noise= att_cfg.POLICY_NOISE,
        noise_clip  = att_cfg.NOISE_CLIP,
        policy_delay= att_cfg.POLICY_DELAY,
        buffer_size = 1000,
    )

    pos_agent = TD3Agent(
        state_dim   = pos_cfg.STATE_DIM,
        action_dim  = pos_cfg.ACTION_DIM,
        max_action  = 1.0,
        hidden_dims = [pos_cfg.HIDDEN_DIMS, pos_cfg.HIDDEN_DIMS],
        actor_lr    = pos_cfg.ACTOR_LR,
        critic_lr   = pos_cfg.CRITIC_LR,
        gamma       = pos_cfg.GAMMA,
        tau         = pos_cfg.TAU,
        policy_noise= pos_cfg.POLICY_NOISE,
        noise_clip  = pos_cfg.NOISE_CLIP,
        policy_delay= pos_cfg.POLICY_DELAY,
        buffer_size = 1000,
    )

    if att_ckpt and os.path.exists(att_ckpt):
        att_agent.load(att_ckpt)
        print(f"    attitude  ← {att_ckpt}")
    else:
        print(f"    attitude  ← random weights (no checkpoint)")

    if pos_ckpt and os.path.exists(pos_ckpt):
        pos_agent.load(pos_ckpt)
        print(f"    position  ← {pos_ckpt}")
    else:
        print(f"    position  ← random weights (no checkpoint)")

    env = PositionEnv(attitude_agent=att_agent)
    env.set_trajectory_scale(1.0)
    env.set_start_radius(0.5)

    return (env, pos_agent), SystemConfig().MASS


def load_integrated(ckpt, agent_type, demo=False):
    """
    Load an integrated single-agent system (td3 or ddpg).
    Returns (env, agent, base_mass).
    """
    if demo:
        return None, None, 1.0

    if INTEGRATED_ROOT not in sys.path:
        sys.path.insert(0, INTEGRATED_ROOT)

    from environments.integrated_env import IntegratedEnv
    from configs.config import IntegratedAgentConfig, SystemConfig

    cfg     = IntegratedAgentConfig()
    sys_cfg = SystemConfig()

    if agent_type == 'td3':
        from agents.td3_agent import TD3Agent
        agent = TD3Agent(
            state_dim    = cfg.STATE_DIM,
            action_dim   = cfg.ACTION_DIM,
            max_action   = cfg.ACTION_SCALE,
            hidden_dims  = cfg.HIDDEN_DIMS,
            actor_lr     = getattr(cfg, 'ACTOR_LR',     3e-5),
            critic_lr    = getattr(cfg, 'CRITIC_LR',    1e-4),
            gamma        = cfg.GAMMA,
            tau          = cfg.TAU,
            policy_noise = getattr(cfg, 'POLICY_NOISE', 0.15),
            noise_clip   = getattr(cfg, 'NOISE_CLIP',   0.3),
            policy_delay = getattr(cfg, 'POLICY_DELAY', 2),
            buffer_size  = 1000,
        )
    else:   # ddpg
        from agents.ddpg_agent import DDPGAgent
        agent = DDPGAgent(
            state_dim         = cfg.STATE_DIM,
            action_dim        = cfg.ACTION_DIM,
            max_action        = cfg.ACTION_SCALE,
            hidden_dims       = cfg.HIDDEN_DIMS,
            actor_lr          = getattr(cfg, 'ACTOR_LR',          5e-5),
            critic_lr         = getattr(cfg, 'CRITIC_LR',         3e-4),
            gamma             = cfg.GAMMA,
            tau               = cfg.TAU,
            target_noise      = getattr(cfg, 'TARGET_NOISE',      0.05),
            target_noise_clip = getattr(cfg, 'TARGET_NOISE_CLIP', 0.1),
            buffer_size       = 1000,
        )

    if ckpt and os.path.exists(ckpt):
        agent.load(ckpt)
        print(f"    agent  ← {ckpt}")
    else:
        print(f"    agent  ← random weights (no checkpoint)")

    env = IntegratedEnv()
    env.set_trajectory_scale(1.0)
    env.set_start_radius(0.5)

    return env, agent, sys_cfg.MASS


# ══════════════════════════════════════════════════════════════════════════════
#  RUN ONE EPISODE
# ══════════════════════════════════════════════════════════════════════════════

def run_cascaded_episode(env_agent, params, base_mass, demo=False):
    """One evaluation episode for the cascaded TD3 system."""
    if demo:
        return _synthetic_episode(base=85, params=params, n_actions=3, worse_frac=1.0)

    env, pos_agent = env_agent
    apply_disturbance(env, params, base_mass)

    sensor_std = params['sensor_noise_std']
    act_noise  = params['actuator_noise']

    state  = env.reset()
    pos_errors, attitudes, actions = [], [], []
    crashed = False
    term    = None

    while True:
        obs    = state + np.random.normal(0, sensor_std, state.shape) if sensor_std > 0 else state
        action = pos_agent.select_action(obs, explore=False)

        if act_noise > 0:
            action = np.clip(action + np.random.normal(0, act_noise, action.shape), -1.0, 1.0)

        next_state, _reward, done, info = env.step(action)

        pos_errors.append(info['pos_error'])
        attitudes.append(np.rad2deg(env.state[6:9]))
        actions.append(action.copy())

        term = info.get('termination', '')
        if term in CRASH_TERMS:
            crashed = True

        if done:
            break
        state = next_state

    env.dynamics.m = base_mass
    return episode_metrics(pos_errors, attitudes, actions, crashed, env.sys_cfg.DT)


def run_integrated_episode(env, agent, params, base_mass, demo=False, demo_base=70):
    """One evaluation episode for an integrated single-agent system."""
    if demo:
        return _synthetic_episode(base=demo_base, params=params, n_actions=4, worse_frac=0.85)

    apply_disturbance(env, params, base_mass)

    sensor_std = params['sensor_noise_std']
    act_noise  = params['actuator_noise']

    state  = env.reset()
    pos_errors, attitudes, actions = [], [], []
    crashed = False
    term    = None

    while True:
        obs    = state + np.random.normal(0, sensor_std, state.shape) if sensor_std > 0 else state
        action = agent.select_action(obs, explore=False)

        if act_noise > 0:
            action = np.clip(action + np.random.normal(0, act_noise, action.shape), -1.0, 1.0)

        next_state, _reward, done, info = env.step(action)

        pos_errors.append(info['pos_error'])
        attitudes.append(np.rad2deg(env.state[6:9]))
        actions.append(action.copy())

        term = info.get('termination', '')
        if term in CRASH_TERMS:
            crashed = True

        if done:
            break
        state = next_state

    env.dynamics.m = base_mass
    return episode_metrics(pos_errors, attitudes, actions, crashed, env.sys_cfg.DT)


def _synthetic_episode(base, params, n_actions, worse_frac):
    """Plausible fake episode for demo / dry-run."""
    scale  = 1.0
    scale += params.get('wind_std',           [0])[0]  * 4000
    scale += params.get('sensor_noise_std',   0)       * 1500
    scale += params.get('mass_noise_frac',    0)       * 400
    scale += params.get('actuator_noise',     0)       * 800
    eff_base = base * worse_frac + (base / worse_frac) * (1 - worse_frac) + scale
    n      = 5000
    errs   = np.abs(np.random.normal(eff_base, eff_base * 0.3, n))
    atts   = np.random.normal(0, 4.5, (n, 3))
    acts   = np.random.uniform(-0.25, 0.25, (n, n_actions))
    crash  = np.random.random() < 0.015 * (1 + params.get('wind_std', [0])[0] * 100)
    return episode_metrics(errs, atts, acts, crash, dt=0.01)


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATE ONE SYSTEM OVER ALL CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_system(label, run_fn, conditions, n_episodes):
    """
    Run n_episodes per condition for the given system.
    run_fn(params) → episode_metrics dict
    Returns {condition_id: aggregate_metrics dict}.
    """
    print(f"\n  ┌─ {label} ─{'─'*max(0, 50-len(label))}┐")
    results = {}
    for cid, cname, params in conditions:
        ep_results = []
        for _ in range(n_episodes):
            ep_results.append(run_fn(params))
        agg = aggregate_metrics(ep_results)
        results[cid] = agg
        print(f"  │  {cname:<28}  RMSE={agg['rmse_mean']:6.1f} mm  "
              f"Succ={agg['success_rate']:5.1f}%  Crash={agg['crash_rate']:4.1f}%")
    print(f"  └{'─'*58}┘")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  TABLE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def fmt(val, spec='.1f', na='—'):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return na
    return format(val, spec)


def print_console_table(conditions, systems):
    """
    systems : list of (label, results_dict)
    """
    n   = len(systems)
    col = 54

    print()
    print('=' * (28 + n * (col + 3)))
    print('PERFORMANCE TABLE')
    print('=' * (28 + n * (col + 3)))

    # System headers
    header = f"{'Condition':<28}"
    for label, _ in systems:
        header += f" │ {label:^52}"
    print(header)

    sub = f"{'':28}"
    metric_hdr = f"{'RMSE':>8} {'Mean±Std':>15} {'Max':>7} {'Succ%':>6} {'Crash%':>7} {'Conv(s)':>7}"
    for _ in systems:
        sub += f" │ {metric_hdr}"
    print(sub)
    print('-' * (28 + n * (col + 3)))

    group_labels = {
        'noisefree': '── Baseline',
        'wind1':     '── Wind Disturbances',
        'sensor':    '── System Noise',
        'combined':  '── Combined',
    }

    for cid, cname, _ in conditions:
        if cid in group_labels:
            print(f"\n  {group_labels[cid]}")

        row = f"{cname:<28}"
        for _, res in systems:
            r  = res[cid]
            ms = f"{fmt(r['mean_err'])}±{fmt(r['mean_err_std'])}"
            row += (f" │ {fmt(r['rmse_mean']):>8} {ms:>15} {fmt(r['max_err']):>7}"
                    f" {fmt(r['success_rate'],'.0f'):>6} {fmt(r['crash_rate'],'.0f'):>7}"
                    f" {fmt(r['conv_time']):>7}")
        print(row)

    print('\n' + '=' * (28 + n * (col + 3)))
    print(f"  RMSE = root-mean-square 3-D pos error (mm)")
    print(f"  Succ% = episodes with RMSE < 300 mm")
    print(f"  Conv  = time to first sustain < 100 mm for ≥ 20 steps (s)\n")


def save_latex(conditions, systems, filepath):
    """systems: list of (label, results_dict)"""
    n_sys   = len(systems)
    n_cols  = n_sys * 6           # 6 metrics per system
    col_fmt = 'l' + (' | ' + 'r' * 6) * n_sys

    group_headers = {
        'noisefree': r'\multicolumn{' + str(1 + n_cols) + r'}{l}{\textit{Baseline}} \\',
        'wind1':     r'\multicolumn{' + str(1 + n_cols) + r'}{l}{\textit{Wind Disturbances}} \\',
        'sensor':    r'\multicolumn{' + str(1 + n_cols) + r'}{l}{\textit{System Noise}} \\',
        'combined':  r'\multicolumn{' + str(1 + n_cols) + r'}{l}{\textit{Combined Conditions}} \\',
    }

    lines = []
    lines.append(r'% Auto-generated by evaluate_performance_table.py')
    lines.append(r'\begin{table*}[!ht]')
    lines.append(r'\centering')
    lines.append(r'\caption{Performance comparison of quadcopter controllers under '
                 r'varied disturbance conditions on a 3-D expanding-helix trajectory '
                 r'($T=50\,$s). Bold values indicate the best result in each column.}')
    lines.append(r'\label{tab:performance}')
    lines.append(r'\small')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(r'\begin{tabular}{' + col_fmt + r'}')
    lines.append(r'\toprule')

    # System name row
    sys_header = ''
    for i, (label, _) in enumerate(systems):
        sys_header += r' & \multicolumn{6}{' + (r'|c' if i == 0 else r'c') + r'}{\textbf{' + label + r'}}'
    lines.append(r'\textbf{Condition}' + sys_header + r' \\')

    # cmidrule
    cmidrule = ''
    for i in range(n_sys):
        c1 = 2 + i * 6
        c2 = c1 + 5
        cmidrule += rf'\cmidrule(lr){{{c1}-{c2}}}'
    lines.append(cmidrule)

    # Metric sub-headers (one per system)
    metric_names = [r'\textbf{RMSE}', r'\textbf{Mean$\pm$Std}', r'\textbf{Max}',
                    r'\textbf{Succ.}', r'\textbf{Crash}', r'\textbf{Conv.}']
    sub_hdr = r'\textbf{Disturbance Condition}' + (' & ' + ' & '.join(metric_names)) * n_sys + r' \\'
    lines.append(sub_hdr)

    units = [r'\textbf{(mm)}', r'\textbf{(mm)}', r'\textbf{(mm)}',
             r'\textbf{(\%)}', r'\textbf{(\%)}', r'\textbf{(s)}']
    unit_row = '' + (' & ' + ' & '.join(units)) * n_sys + r' \\'
    lines.append(unit_row)
    lines.append(r'\midrule')

    def best_val(cid, col_fn, lower_is_better=True):
        """Return the best (index, value) across all systems for a metric."""
        vals = []
        for _, res in systems:
            try:
                v = col_fn(res[cid])
                vals.append(float(v) if v != '—' else float('inf'))
            except Exception:
                vals.append(float('inf'))
        best_idx = int(np.argmin(vals)) if lower_is_better else int(np.argmax(vals))
        return best_idx

    for cid, cname, _ in conditions:
        if cid in group_headers:
            lines.append(group_headers[cid])

        # Determine which system is best for each metric
        best_rmse  = best_val(cid, lambda r: r['rmse_mean'],    lower_is_better=True)
        best_succ  = best_val(cid, lambda r: r['success_rate'], lower_is_better=False)
        best_crash = best_val(cid, lambda r: r['crash_rate'],   lower_is_better=True)
        best_conv  = best_val(cid, lambda r: r['conv_time'],    lower_is_better=True)

        row = f'  {cname}'
        for si, (_, res) in enumerate(systems):
            r  = res[cid]
            ms = f"{fmt(r['mean_err'])}$\\pm${fmt(r['mean_err_std'])}"

            rmse_s  = fmt(r['rmse_mean'])
            succ_s  = fmt(r['success_rate'], '.0f')
            crash_s = fmt(r['crash_rate'],   '.0f')
            conv_s  = fmt(r['conv_time'])

            if si == best_rmse:  rmse_s  = rf'\textbf{{{rmse_s}}}'
            if si == best_succ:  succ_s  = rf'\textbf{{{succ_s}}}'
            if si == best_crash: crash_s = rf'\textbf{{{crash_s}}}'
            if si == best_conv:  conv_s  = rf'\textbf{{{conv_s}}}'

            row += (f' & {rmse_s} & {ms} & {fmt(r["max_err"])}'
                    f' & {succ_s} & {crash_s} & {conv_s}')
        row += r' \\'
        lines.append(row)

    # Summary row
    lines.append(r'\midrule')
    all_row = r'  \textbf{Mean (all conditions)}'
    for si, (_, res) in enumerate(systems):
        all_rmse = np.mean([res[cid]['rmse_mean']    for cid,_,_ in conditions])
        all_succ = np.mean([res[cid]['success_rate'] for cid,_,_ in conditions])
        all_row += f' & {fmt(all_rmse)} & — & — & {fmt(all_succ, ".0f")} & — & —'
    all_row += r' \\'
    lines.append(all_row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\vspace{0.4em}')
    lines.append(r'\begin{flushleft}\footnotesize')
    lines.append(
        r'RMSE: root-mean-square 3-D position error; '
        r'Mean$\pm$Std: mean $\pm$ standard deviation of per-step position error; '
        r'Max: mean peak per-episode error; '
        r'Succ.: episodes with RMSE\,$<$\,300\,mm (\%); '
        r'Crash: episodes ending in crash or bounds violation (\%); '
        r'Conv.: mean time to first sustain $<$100\,mm error (s). '
        r'\textbf{Bold} = best result per column.'
    )
    lines.append(r'\end{flushleft}')
    lines.append(r'\end{table*}')

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  LaTeX  → {filepath}")


def save_csv(conditions, systems, filepath):
    cols = ['condition', 'system',
            'rmse_mm', 'rmse_std_mm', 'mean_mm', 'mean_std_mm', 'max_mm',
            'success_pct', 'crash_pct', 'conv_time_s', 'smoothness',
            'mean_roll_deg', 'mean_pitch_deg']
    rows = []
    for cid, cname, _ in conditions:
        for label, res in systems:
            r = res[cid]
            rows.append([
                cname, label,
                fmt(r['rmse_mean'],     '.3f'),
                fmt(r['rmse_std'],      '.3f'),
                fmt(r['mean_err'],      '.3f'),
                fmt(r['mean_err_std'],  '.3f'),
                fmt(r['max_err'],       '.3f'),
                fmt(r['success_rate'],  '.1f'),
                fmt(r['crash_rate'],    '.1f'),
                fmt(r['conv_time'],     '.3f'),
                fmt(r['smoothness'],    '.5f'),
                fmt(r['mean_roll'],     '.2f'),
                fmt(r['mean_pitch'],    '.2f'),
            ])
    with open(filepath, 'w', newline='') as f:
        csv.writer(f).writerows([cols] + rows)
    print(f"  CSV    → {filepath}")


def save_npy(conditions, systems, filepath):
    data = {
        'systems':    [(lbl, res) for lbl, res in systems],
        'conditions': [(cid, cname) for cid, cname, _ in conditions],
    }
    np.save(filepath, data, allow_pickle=True)
    print(f"  NumPy  → {filepath}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Cascaded TD3 / Integrated TD3 / Integrated DDPG performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Checkpoint paths (all optional) ──────────────────────────────────────
    parser.add_argument('--cascaded_att',    default=None,
                        metavar='PATH',
                        help='Cascaded TD3: attitude agent checkpoint (.pth)')
    parser.add_argument('--cascaded_pos',    default=None,
                        metavar='PATH',
                        help='Cascaded TD3: position agent checkpoint (.pth)')
    parser.add_argument('--integrated_td3',  default=None,
                        metavar='PATH',
                        help='Integrated TD3: single agent checkpoint (.pth)')
    parser.add_argument('--integrated_ddpg', default=None,
                        metavar='PATH',
                        help='Integrated DDPG: single agent checkpoint (.pth)')

    # ── Evaluation settings ───────────────────────────────────────────────────
    parser.add_argument('--episodes', type=int, default=30,
                        help='Evaluation episodes per condition (default: 30)')
    parser.add_argument('--output',   default='results/',
                        help='Output directory (default: results/)')
    parser.add_argument('--demo',     action='store_true',
                        help='Demo mode: random-weight agents, synthetic episode data')
    parser.add_argument('--seed',     type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()
    np.random.seed(args.seed)

    # ── Decide which systems to evaluate ─────────────────────────────────────
    want_cascaded    = args.cascaded_att   or args.cascaded_pos
    want_td3_int     = args.integrated_td3
    want_ddpg_int    = args.integrated_ddpg

    # If nothing is requested, fall into demo mode
    if not (want_cascaded or want_td3_int or want_ddpg_int):
        print('\n  [INFO] No checkpoint paths supplied — switching to --demo mode.')
        print('         Pass --cascaded_att / --cascaded_pos / --integrated_td3 /')
        print('         --integrated_ddpg to evaluate real trained agents.\n')
        args.demo = True
        # In demo mode, evaluate all three so the table is always complete
        want_cascaded = want_td3_int = want_ddpg_int = True

    os.makedirs(args.output, exist_ok=True)
    conditions = make_conditions()

    print('\n' + '=' * 65)
    print('  QUADCOPTER CONTROLLER — PERFORMANCE TABLE EVALUATION')
    print('=' * 65)
    print(f'  Mode     : {"DEMO (synthetic)" if args.demo else "REAL"}')
    print(f'  Episodes : {args.episodes} per condition')
    print(f'  Seed     : {args.seed}')
    print(f'  Output   : {args.output}')
    systems_to_run = (
        (['Cascaded TD3']     if want_cascaded else []) +
        (['Integrated TD3']   if want_td3_int  else []) +
        (['Integrated DDPG']  if want_ddpg_int else [])
    )
    print(f'  Systems  : {", ".join(systems_to_run)}')
    print('=' * 65)

    # ── Load systems ──────────────────────────────────────────────────────────
    evaluated_systems = []   # list of (label, results_dict)

    # 1. Cascaded TD3
    if want_cascaded:
        print('\n[1/3] Loading Cascaded TD3...')
        casc, base_mass_casc = load_cascaded_td3(
            args.cascaded_att, args.cascaded_pos, demo=args.demo)

        def run_casc(params):
            return run_cascaded_episode(casc, params, base_mass_casc, demo=args.demo)

        t0 = time.time()
        res_casc = evaluate_system('Cascaded TD3', run_casc, conditions, args.episodes)
        print(f'    ✓ done in {time.time()-t0:.1f}s')
        evaluated_systems.append(('Cascaded TD3', res_casc))

    # 2. Integrated TD3
    if want_td3_int:
        print('\n[2/3] Loading Integrated TD3...')
        env_td3, agent_td3, base_mass_td3 = load_integrated(
            args.integrated_td3, 'td3', demo=args.demo)

        def run_td3(params):
            return run_integrated_episode(
                env_td3, agent_td3, params, base_mass_td3,
                demo=args.demo, demo_base=75)

        t0 = time.time()
        res_td3 = evaluate_system('Integrated TD3', run_td3, conditions, args.episodes)
        print(f'    ✓ done in {time.time()-t0:.1f}s')
        evaluated_systems.append(('Integrated TD3', res_td3))

    # 3. Integrated DDPG
    if want_ddpg_int:
        print('\n[3/3] Loading Integrated DDPG...')
        env_ddpg, agent_ddpg, base_mass_ddpg = load_integrated(
            args.integrated_ddpg, 'ddpg', demo=args.demo)

        def run_ddpg(params):
            return run_integrated_episode(
                env_ddpg, agent_ddpg, params, base_mass_ddpg,
                demo=args.demo, demo_base=70)

        t0 = time.time()
        res_ddpg = evaluate_system('Integrated DDPG', run_ddpg, conditions, args.episodes)
        print(f'    ✓ done in {time.time()-t0:.1f}s')
        evaluated_systems.append(('Integrated DDPG', res_ddpg))

    # ── Print & save ──────────────────────────────────────────────────────────
    print_console_table(conditions, evaluated_systems)

    print('Saving outputs...')
    save_latex(conditions, evaluated_systems,
               os.path.join(args.output, 'performance_table.tex'))
    save_csv(conditions, evaluated_systems,
             os.path.join(args.output, 'performance_results.csv'))
    save_npy(conditions, evaluated_systems,
             os.path.join(args.output, 'performance_results.npy'))

    print(f'\n  Output files:')
    print(f'    performance_table.tex   ← paste into paper')
    print(f'    performance_results.csv ← spreadsheet / pandas')
    print(f'    performance_results.npy ← for plotting scripts')
    print()


if __name__ == '__main__':
    main()
