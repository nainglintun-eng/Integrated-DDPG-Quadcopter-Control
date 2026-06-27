"""
Trajectory Visualization – Integrated Single-Agent TD3
=======================================================
Creates:
  1. Animated GIF / MP4 – real-time 3-D tracking from multiple random starts
  2. Multi-start comparison – full performance breakdown across N episodes
  3. Control analysis  – Ft, tau_x/y/z, reward components per episode

Differences from the cascaded version
--------------------------------------
  - Loads ONE agent (not attitude + position separately)
  - Action vector is [Ft, tau_x, tau_y, tau_z] → dedicated control subplot
  - No att_des / system_solve intermediate variables to display
  - info dict keys: 'Ft', 'torques' (shape 3), 'pos_error' (mm), 'vel_error' (m/s)
  - State layout: [x,y,z, u,v,w, phi,theta,psi, p,q,r]

Usage
-----
  # Multi-start comparison + animation:
  python trajectory_visualization.py \
      --agent results/run_XYZ/integrated_controller_final.pth \
      --multi_start --video

  # Just the animation:
  python trajectory_visualization.py \
      --agent results/run_XYZ/integrated_phase_1_best.pth \
      --video --video_name helix.gif --fps 25

  # Quick sanity check (random weights, no checkpoint needed):
  python trajectory_visualization.py --multi_start --num_starts 4
"""

import argparse
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')           # headless-safe; swap to 'TkAgg' for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers 3d projection)
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.integrated_env import IntegratedEnv
#from agents.td3_agent             import TD3Agent
from agents.ddpg_agent import DDPGAgent
from configs.config               import (SystemConfig,
                                          IntegratedAgentConfig,
                                          get_trajectory_function)


# ──────────────────────────────────────────────────────────────────────────────
#  Episode runner
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(env, agent, start_pos=None, max_time=None, wind_std=None,
                explore=False):
    """
    Roll out one episode with the integrated agent.

    Returns a dict of time-series arrays:
        time          (T,)
        pos_actual    (T, 3)   world-frame  [x, y, z]
        pos_desired   (T, 3)
        vel_actual    (T, 3)   body-frame   [u, v, w]
        vel_desired   (T, 3)
        attitude_deg  (T, 3)   [phi, theta, psi]  in degrees
        rates_deg     (T, 3)   [p, q, r]           in degrees/s
        Ft            (T,)     total thrust  (N)
        tau           (T, 3)   [tau_x, tau_y, tau_z]  (Nm)
        action_norm   (T, 4)   raw normalised actor output  [-1, +1]
        pos_error_m   (T,)     position error  (m)
        pos_error_mm  (T,)     position error  (mm)
        vel_error     (T,)     velocity error  (m/s)
        reward        (T,)
        reward_components  dict of (T,) arrays
        start_pos     (3,)
        termination   str
    """
    if wind_std is not None:
        env.set_wind(wind_std)

    if max_time is not None:
        _orig = env.max_steps
        env.max_steps = int(max_time / env.sys_cfg.DT)

    obs = env.reset(initial_pos=start_pos)
    start_pos_rec = env.state[0:3].copy()

    buf = dict(
        time=[], pos_actual=[], pos_desired=[],
        vel_actual=[], vel_desired=[],
        attitude_deg=[], rates_deg=[],
        Ft=[], tau=[], action_norm=[],
        pos_error_m=[], pos_error_mm=[], vel_error=[], reward=[],
        # reward component buffers filled lazily
        _rc={}
    )

    done = False
    while not done:
        action = agent.select_action(obs, explore=explore)
        obs, reward, done, info = env.step(action)

        buf['time'].append(env.time)
        buf['pos_actual'].append(env.state[0:3].copy())
        buf['vel_actual'].append(env.state[3:6].copy())
        buf['attitude_deg'].append(np.rad2deg(env.state[6:9].copy()))
        buf['rates_deg'].append(np.rad2deg(env.state[9:12].copy()))

        pos_des, vel_des, _ = env.get_trajectory(env.time, env.trajectory_scale)
        buf['pos_desired'].append(pos_des.copy())
        buf['vel_desired'].append(vel_des.copy())

        buf['Ft'].append(float(info['Ft']))
        buf['tau'].append(info['torques'].copy())
        buf['action_norm'].append(action.copy())

        buf['pos_error_m'].append(info['pos_error_m'])
        buf['pos_error_mm'].append(info['pos_error'])
        buf['vel_error'].append(info['vel_error'])
        buf['reward'].append(reward)

        for k, v in info['reward_components'].items():
            buf['_rc'].setdefault(k, []).append(float(v))

    termination = info.get('termination', 'unknown')

    if max_time is not None:
        env.max_steps = _orig

    # Convert to arrays
    data = {k: np.array(v) for k, v in buf.items()
            if k not in ('_rc',)}
    data['reward_components'] = {k: np.array(v) for k, v in buf['_rc'].items()}
    data['start_pos']   = start_pos_rec
    data['termination'] = termination
    return data


# ──────────────────────────────────────────────────────────────────────────────
#  Reference trajectory helper
# ──────────────────────────────────────────────────────────────────────────────

def _ref_traj(env, duration, scale=1.0, n=600):
    traj_fn = get_trajectory_function()
    t = np.linspace(0, duration, n)
    pts = np.array([traj_fn(ti, scale)[0] for ti in t])
    return t, pts


# ──────────────────────────────────────────────────────────────────────────────
#  1.  Animated video
# ──────────────────────────────────────────────────────────────────────────────

def create_animated_video(env, agent, save_path='trajectory.gif',
                          num_starts=3, start_radius=3.0,
                          fps=20, duration=50.0):
    """
    Animated 6-panel video:
      Panel 1 (large): 3-D trajectory
      Panel 2: XY top-down view
      Panel 3: Altitude vs time
      Panel 4: Position error vs time
      Panel 5: Thrust Ft vs time
      Panel 6: Torques tau_x/y/z vs time
    """
    print('\n' + '='*70)
    print('CREATING ANIMATED VIDEO  –  INTEGRATED AGENT')
    print('='*70)

    sc = env.sys_cfg

    # ── Random start positions ──
    np.random.seed(42)
    starts = []
    for i in range(num_starts):
        ang  = np.random.uniform(0, 2*np.pi)
        rad  = np.random.uniform(1.5, start_radius)
        z    = np.random.uniform(0, min(1.5, start_radius * 0.3))
        starts.append(np.array([rad*np.cos(ang), rad*np.sin(ang), z]))
        print(f'  Start {i+1}: ({starts[-1][0]:6.2f}, {starts[-1][1]:6.2f}, '
              f'{starts[-1][2]:5.2f})  |'
              f'  dist_xy={np.linalg.norm(starts[-1][:2]):.2f} m')

    # ── Run episodes ──
    print(f'\nRunning {num_starts} episodes...')
    episodes = []
    for i, sp in enumerate(starts):
        ep = run_episode(env, agent, start_pos=sp, max_time=duration)
        episodes.append(ep)
        print(f'  Ep {i+1}: {len(ep["time"])} steps | '
              f'mean_err={np.mean(ep["pos_error_mm"]):.0f} mm | '
              f'term={ep["termination"]}')

    # ── Reference trajectory ──
    t_ref, traj_ref = _ref_traj(env, duration)

    # ── Figure layout ──
    fig = plt.figure(figsize=(20, 11), facecolor='#1a1a2e')
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            left=0.05, right=0.97,
                            top=0.93, bottom=0.07,
                            hspace=0.38, wspace=0.35)

    ax3d  = fig.add_subplot(gs[:, 0:2], projection='3d')
    ax_xy = fig.add_subplot(gs[0, 2])
    ax_z  = fig.add_subplot(gs[0, 3])
    ax_er = fig.add_subplot(gs[1, 2])
    ax_ft = fig.add_subplot(gs[1, 3])

    _style_ax = dict(facecolor='#0d0d1a')
    for ax in [ax_xy, ax_z, ax_er, ax_ft]:
        ax.set_facecolor('#0d0d1a')
        for spine in ax.spines.values():
            spine.set_color('#555577')
        ax.tick_params(colors='#aaaacc', labelsize=8)
        ax.xaxis.label.set_color('#aaaacc')
        ax.yaxis.label.set_color('#aaaacc')
        ax.title.set_color('#ddddff')
        ax.grid(True, alpha=0.18, color='#555577')

    ax3d.set_facecolor('#0d0d1a')
    ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.label.set_color('#aaaacc')
        axis.pane.set_edgecolor('#333355')
    ax3d.tick_params(colors='#aaaacc', labelsize=7)

    # Static reference plots
    ax3d.plot(traj_ref[:, 0], traj_ref[:, 1], traj_ref[:, 2],
              '--', color='#eeeeaa', lw=2, alpha=0.5, label='Desired')
    ax_xy.plot(traj_ref[:, 0], traj_ref[:, 1],
               '--', color='#eeeeaa', lw=2, alpha=0.5)
    ax_z.plot(t_ref, traj_ref[:, 2],
              '--', color='#eeeeaa', lw=2, alpha=0.5, label='Desired')

    # Origin markers
    ax3d.scatter([0], [0], [0], c='gold', s=350, marker='*',
                 edgecolors='black', linewidths=1.5, zorder=100)
    ax_xy.scatter([0], [0], c='gold', s=250, marker='*',
                  edgecolors='black', linewidths=1.5, zorder=100)

    # ── Line handles ──
    colors = plt.cm.plasma(np.linspace(0.15, 0.9, num_starts))

    ln3d, mk3d  = [], []
    ln_xy, mk_xy = [], []
    ln_z, ln_er, ln_ft = [], [], []

    for i, c in enumerate(colors):
        l3,  = ax3d.plot([], [], [], color=c, lw=2.2, alpha=0.85,
                         label=f'Drone {i+1}')
        m3   = ax3d.scatter([], [], [], c=[c], s=180,
                            marker='o', edgecolors='white', linewidths=1.2)
        lxy, = ax_xy.plot([], [], color=c, lw=2, alpha=0.85)
        mxy  = ax_xy.scatter([], [], c=[c], s=120,
                             marker='o', edgecolors='white', linewidths=1.2)
        lz,  = ax_z.plot([], [], color=c, lw=2, alpha=0.85)
        ler, = ax_er.plot([], [], color=c, lw=2, alpha=0.85)
        lft, = ax_ft.plot([], [], color=c, lw=2, alpha=0.85)

        ln3d.append(l3);  mk3d.append(m3)
        ln_xy.append(lxy); mk_xy.append(mxy)
        ln_z.append(lz);  ln_er.append(ler); ln_ft.append(lft)

    # ── Axes limits & labels ──
    max_r = max(np.max(np.abs(traj_ref[:, :2])), start_radius) + 2
    ax3d.set_xlim(-max_r, max_r);  ax3d.set_ylim(-max_r, max_r)
    ax3d.set_zlim(0, traj_ref[:, 2].max() + 5)
    ax3d.set_xlabel('X (m)', labelpad=4); ax3d.set_ylabel('Y (m)', labelpad=4)
    ax3d.set_zlabel('Z (m)', labelpad=4)
    ax3d.set_title('3-D Trajectory Tracking', fontsize=13, color='#ddddff',
                   fontweight='bold', pad=8)
    ax3d.legend(loc='upper left', fontsize=8,
                facecolor='#1a1a2e', labelcolor='#ddddff', framealpha=0.6)
    ax3d.view_init(elev=22, azim=40)

    ax_xy.set_xlim(-max_r, max_r); ax_xy.set_ylim(-max_r, max_r)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlabel('X (m)', fontsize=9); ax_xy.set_ylabel('Y (m)', fontsize=9)
    ax_xy.set_title('Top View (XY)', fontsize=10, fontweight='bold')
    circ = plt.Circle((0, 0), start_radius, fill=False,
                       color='#888899', ls=':', lw=1.5, alpha=0.5)
    ax_xy.add_patch(circ)

    ax_z.set_xlim(0, duration); ax_z.set_ylim(0, traj_ref[:, 2].max() + 5)
    ax_z.set_xlabel('Time (s)', fontsize=9); ax_z.set_ylabel('Altitude (m)', fontsize=9)
    ax_z.set_title('Altitude', fontsize=10, fontweight='bold')

    ax_er.set_xlim(0, duration); ax_er.set_ylim(0, 5)
    ax_er.axhline(0.5, color='#44ff88', ls='--', lw=1.2, alpha=0.6, label='0.5 m')
    ax_er.axhline(1.0, color='#ffaa44', ls='--', lw=1.2, alpha=0.6, label='1.0 m')
    ax_er.set_xlabel('Time (s)', fontsize=9)
    ax_er.set_ylabel('Pos error (m)', fontsize=9)
    ax_er.set_title('Tracking Error', fontsize=10, fontweight='bold')
    ax_er.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='#ddddff',
                 framealpha=0.5)

    ax_ft.set_xlim(0, duration)
    ax_ft.set_ylim(0, sc.MAX_THRUST * 1.05)
    ax_ft.axhline(sc.MASS * sc.GRAVITY, color='#aaffaa', ls='--',
                  lw=1.2, alpha=0.7, label=f'hover {sc.MASS*sc.GRAVITY:.1f} N')
    ax_ft.set_xlabel('Time (s)', fontsize=9)
    ax_ft.set_ylabel('Thrust Ft (N)', fontsize=9)
    ax_ft.set_title('Thrust', fontsize=10, fontweight='bold')
    ax_ft.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='#ddddff',
                 framealpha=0.5)

    # Title + time stamp
    fig.suptitle('Integrated DDPG – Quadcopter Trajectory Tracking',
                 fontsize=14, color='white', fontweight='bold')
    time_txt = ax3d.text2D(0.02, 0.97, '', transform=ax3d.transAxes,
                           fontsize=12, fontweight='bold', color='white',
                           va='top',
                           bbox=dict(boxstyle='round', fc='#1a1a2e',
                                     ec='#6666aa', alpha=0.8))

    # ── Animation ──
    dt         = sc.DT
    frame_skip = max(1, int(1 / (fps * dt)))
    max_frames = min(len(ep['time']) for ep in episodes)
    frame_idxs = list(range(0, max_frames, frame_skip))

    def _init():
        for i in range(num_starts):
            ln3d[i].set_data([], []); ln3d[i].set_3d_properties([])
            mk3d[i]._offsets3d = ([], [], [])
            ln_xy[i].set_data([], [])
            mk_xy[i].set_offsets(np.empty((0, 2)))
            ln_z[i].set_data([], [])
            ln_er[i].set_data([], [])
            ln_ft[i].set_data([], [])
        time_txt.set_text('')
        return ln3d + mk3d + ln_xy + mk_xy + ln_z + ln_er + ln_ft + [time_txt]

    def _update(frame_num):
        idx = frame_idxs[frame_num]
        for i, ep in enumerate(episodes):
            n = min(idx, len(ep['time']) - 1)
            p = ep['pos_actual']

            ln3d[i].set_data(p[:n, 0], p[:n, 1])
            ln3d[i].set_3d_properties(p[:n, 2])
            mk3d[i]._offsets3d = ([p[n, 0]], [p[n, 1]], [p[n, 2]])

            ln_xy[i].set_data(p[:n, 0], p[:n, 1])
            mk_xy[i].set_offsets([[p[n, 0], p[n, 1]]])

            ln_z[i].set_data(ep['time'][:n], p[:n, 2])
            ln_er[i].set_data(ep['time'][:n], ep['pos_error_m'][:n])
            ln_ft[i].set_data(ep['time'][:n], ep['Ft'][:n])

        t_cur = episodes[0]['time'][min(idx, len(episodes[0]['time']) - 1)]
        time_txt.set_text(f't = {t_cur:.1f} s')
        return ln3d + mk3d + ln_xy + mk_xy + ln_z + ln_er + ln_ft + [time_txt]

    anim = FuncAnimation(fig, _update, init_func=_init,
                         frames=len(frame_idxs),
                         interval=1000 / fps, blit=True)

    print(f'\nSaving → {save_path}  ({len(frame_idxs)} frames @ {fps} fps)...')
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if save_path.endswith('.mp4'):
        anim.save(save_path, writer='ffmpeg', fps=fps, dpi=130,
                  savefig_kwargs={'facecolor': '#1a1a2e'})
    else:
        anim.save(save_path, writer=PillowWriter(fps=fps), dpi=100,
                  savefig_kwargs={'facecolor': '#1a1a2e'})

    plt.close(fig)
    print(f'✓ Animation saved.')
    return episodes


# ──────────────────────────────────────────────────────────────────────────────
#  2.  Multi-start comparison  (static, high-res)
# ──────────────────────────────────────────────────────────────────────────────

def plot_multi_start_comparison(env, agent, num_starts=8,
                                start_radius=5.0, save_dir='results/visualization'):
    """
    10-panel static figure:
      Row 1: 3-D traj | XY | altitude | pos error | vel error
      Row 2: roll/pitch | yaw | Ft | torques | stats table
    """
    os.makedirs(save_dir, exist_ok=True)
    sc = env.sys_cfg

    print('\n' + '='*70)
    print(f'MULTI-START COMPARISON  –  {num_starts} starts')
    print('='*70)

    # ── Generate starts ──
    np.random.seed(0)
    starts = []
    for i in range(num_starts):
        ang = np.random.uniform(0, 2*np.pi)
        rad = np.random.uniform(1.0, start_radius)
        z   = np.random.uniform(0, min(1.5, start_radius * 0.25))
        starts.append(np.array([rad*np.cos(ang), rad*np.sin(ang), z]))
        print(f'  Start {i+1:2d}: ({starts[-1][0]:6.2f}, {starts[-1][1]:6.2f}, '
              f'{starts[-1][2]:4.2f})')

    # ── Run episodes ──
    print('\nRunning episodes...')
    episodes = []
    for i, sp in enumerate(starts):
        ep = run_episode(env, agent, start_pos=sp)
        episodes.append(ep)
        print(f'  Ep {i+1:2d}: {len(ep["time"]):5d} steps | '
              f'mean={np.mean(ep["pos_error_mm"]):6.0f} mm | '
              f'final={ep["pos_error_mm"][-1]:6.0f} mm | '
              f'term={ep["termination"]}')

    max_time = max(ep['time'][-1] for ep in episodes)
    t_ref, traj_ref = _ref_traj(env, max_time)
    colors  = plt.cm.tab10(np.linspace(0, 1, num_starts))

    # ── Figure ──
    fig = plt.figure(figsize=(24, 13), facecolor='white')
    gs  = gridspec.GridSpec(2, 5, figure=fig,
                            left=0.05, right=0.98,
                            top=0.92, bottom=0.07,
                            hspace=0.42, wspace=0.35)

    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    axs  = [
        fig.add_subplot(gs[0, 1]),   # XY
        fig.add_subplot(gs[0, 2]),   # altitude
        fig.add_subplot(gs[0, 3]),   # pos error
        fig.add_subplot(gs[0, 4]),   # vel error
        fig.add_subplot(gs[1, 1]),   # roll/pitch
        fig.add_subplot(gs[1, 2]),   # yaw
        fig.add_subplot(gs[1, 3]),   # Ft
        fig.add_subplot(gs[1, 4]),   # torques (tau_x only shown, others dashed)
    ]
    ax_xy, ax_alt, ax_er, ax_ver, \
        ax_rp, ax_yaw, ax_ft, ax_tau = axs

    # ── 3-D ──
    ax3d.plot(traj_ref[:, 0], traj_ref[:, 1], traj_ref[:, 2],
              'k--', lw=2.5, alpha=0.4, label='Desired')
    for i, (ep, c) in enumerate(zip(episodes, colors)):
        p = ep['pos_actual']
        ax3d.plot(p[:, 0], p[:, 1], p[:, 2], color=c, lw=1.8, alpha=0.8)
        ax3d.scatter(*ep['start_pos'], c=[c], s=120,
                     marker='o', edgecolors='k', linewidths=1.5, zorder=10)
        ax3d.scatter(*p[-1], c=[c], s=120,
                     marker='s', edgecolors='k', linewidths=1.5, zorder=10)
    ax3d.scatter([0], [0], [0], c='gold', s=400, marker='*',
                 edgecolors='k', linewidths=2, zorder=20, label='Origin')
    max_r = max(np.max(np.abs(traj_ref[:, :2])), start_radius) + 1.5
    ax3d.set_xlim(-max_r, max_r); ax3d.set_ylim(-max_r, max_r)
    ax3d.set_zlim(0, traj_ref[:, 2].max() + 5)
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3-D Trajectories\n(●=start, ■=end)',
                   fontweight='bold', fontsize=11)
    ax3d.legend(fontsize=8, loc='upper left')
    ax3d.view_init(elev=24, azim=42)
    ax3d.grid(True, alpha=0.25)

    # ── XY ──
    ax_xy.plot(traj_ref[:, 0], traj_ref[:, 1], 'k--', lw=2, alpha=0.35)
    for i, (ep, c) in enumerate(zip(episodes, colors)):
        ax_xy.plot(ep['pos_actual'][:, 0], ep['pos_actual'][:, 1],
                   color=c, lw=1.8, alpha=0.8, label=f'S{i+1}')
        ax_xy.scatter(*ep['start_pos'][:2], c=[c], s=80,
                      marker='o', edgecolors='k', lw=1.2, zorder=5)
    ax_xy.scatter([0], [0], c='gold', s=250, marker='*',
                  edgecolors='k', lw=2, zorder=10)
    ax_xy.add_patch(plt.Circle((0, 0), start_radius, fill=False,
                                color='gray', ls='--', lw=1.5, alpha=0.5))
    ax_xy.set_aspect('equal'); ax_xy.set_xlabel('X (m)'); ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('Top View (XY)', fontweight='bold', fontsize=11)
    ax_xy.legend(fontsize=6, ncol=2, loc='upper right')
    ax_xy.grid(True, alpha=0.3)

    # ── Altitude ──
    ax_alt.plot(t_ref, traj_ref[:, 2], 'k--', lw=2, alpha=0.35, label='Desired')
    for ep, c in zip(episodes, colors):
        ax_alt.plot(ep['time'], ep['pos_actual'][:, 2], color=c, lw=1.5, alpha=0.8)
    ax_alt.set_xlabel('Time (s)'); ax_alt.set_ylabel('Altitude (m)')
    ax_alt.set_title('Altitude Tracking', fontweight='bold', fontsize=11)
    ax_alt.legend(fontsize=8); ax_alt.grid(True, alpha=0.3)

    # ── Position error ──
    all_errs = np.concatenate([ep['pos_error_m'] for ep in episodes])
    y_top = min(float(np.percentile(all_errs, 97)) * 1.2 + 0.2, 8.0)
    ax_er.axhline(0.5, color='green',  ls='--', lw=1.5, alpha=0.6, label='0.5 m')
    ax_er.axhline(1.0, color='orange', ls='--', lw=1.5, alpha=0.6, label='1.0 m')
    for ep, c in zip(episodes, colors):
        ax_er.plot(ep['time'], ep['pos_error_m'], color=c, lw=1.5, alpha=0.8)
    ax_er.set_ylim(0, y_top)
    ax_er.set_xlabel('Time (s)'); ax_er.set_ylabel('Error (m)')
    ax_er.set_title('Position Error', fontweight='bold', fontsize=11)
    ax_er.legend(fontsize=8); ax_er.grid(True, alpha=0.3)

    # ── Velocity error ──
    for ep, c in zip(episodes, colors):
        ax_ver.plot(ep['time'], ep['vel_error'], color=c, lw=1.5, alpha=0.8)
    ax_ver.set_xlabel('Time (s)'); ax_ver.set_ylabel('Vel error (m/s)')
    ax_ver.set_title('Velocity Error', fontweight='bold', fontsize=11)
    ax_ver.grid(True, alpha=0.3)

    # ── Roll / Pitch ──
    lim_deg = np.rad2deg(sc.CRASH_ANGLE) * 0.6
    ax_rp.axhline(0, color='k', lw=0.8, alpha=0.4)
    ax_rp.axhline( lim_deg, color='red', ls=':', lw=1.5, alpha=0.5, label=f'±{lim_deg:.0f}°')
    ax_rp.axhline(-lim_deg, color='red', ls=':', lw=1.5, alpha=0.5)
    for ep, c in zip(episodes, colors):
        ax_rp.plot(ep['time'], ep['attitude_deg'][:, 0],
                   color=c, lw=1.3, ls='-',  alpha=0.75)   # roll  solid
        ax_rp.plot(ep['time'], ep['attitude_deg'][:, 1],
                   color=c, lw=1.3, ls='--', alpha=0.75)   # pitch dashed
    ax_rp.set_xlabel('Time (s)'); ax_rp.set_ylabel('Angle (°)')
    ax_rp.set_title('Roll (—) / Pitch (--)', fontweight='bold', fontsize=11)
    ax_rp.legend(fontsize=8); ax_rp.grid(True, alpha=0.3)
    ax_rp.set_ylim(-lim_deg * 1.4, lim_deg * 1.4)

    # ── Yaw ──
    for ep, c in zip(episodes, colors):
        ax_yaw.plot(ep['time'], ep['attitude_deg'][:, 2],
                    color=c, lw=1.3, alpha=0.8)
    ax_yaw.set_xlabel('Time (s)'); ax_yaw.set_ylabel('Yaw (°)')
    ax_yaw.set_title('Yaw', fontweight='bold', fontsize=11)
    ax_yaw.grid(True, alpha=0.3)

    # ── Thrust ──
    ax_ft.axhline(sc.MASS * sc.GRAVITY, color='green', ls='--',
                  lw=1.5, alpha=0.6,
                  label=f'hover = {sc.MASS*sc.GRAVITY:.1f} N')
    ax_ft.axhline(sc.MAX_THRUST, color='red', ls=':',
                  lw=1.5, alpha=0.5, label=f'max = {sc.MAX_THRUST:.1f} N')
    for ep, c in zip(episodes, colors):
        ax_ft.plot(ep['time'], ep['Ft'], color=c, lw=1.3, alpha=0.8)
    ax_ft.set_ylim(0, sc.MAX_THRUST * 1.05)
    ax_ft.set_xlabel('Time (s)'); ax_ft.set_ylabel('Ft (N)')
    ax_ft.set_title('Total Thrust', fontweight='bold', fontsize=11)
    ax_ft.legend(fontsize=8); ax_ft.grid(True, alpha=0.3)

    # ── Torques ──
    ax_tau.axhline( sc.MAX_TORQUE,     color='red', ls=':', lw=1, alpha=0.4)
    ax_tau.axhline(-sc.MAX_TORQUE,     color='red', ls=':', lw=1, alpha=0.4,
                   label=f'±τ_xy max = {sc.MAX_TORQUE:.2f} Nm')
    ax_tau.axhline( sc.MAX_TORQUE_YAW, color='orange', ls=':', lw=1, alpha=0.4)
    ax_tau.axhline(-sc.MAX_TORQUE_YAW, color='orange', ls=':', lw=1, alpha=0.4,
                   label=f'±τ_z max = {sc.MAX_TORQUE_YAW:.2f} Nm')
    for ep, c in zip(episodes, colors):
        ax_tau.plot(ep['time'], ep['tau'][:, 0],
                    color=c, lw=1.1, ls='-',  alpha=0.65)   # tau_x solid
        ax_tau.plot(ep['time'], ep['tau'][:, 1],
                    color=c, lw=1.1, ls='--', alpha=0.65)   # tau_y dashed
        ax_tau.plot(ep['time'], ep['tau'][:, 2],
                    color=c, lw=1.1, ls=':',  alpha=0.65)   # tau_z dotted
    ax_tau.set_xlabel('Time (s)'); ax_tau.set_ylabel('Torque (Nm)')
    ax_tau.set_title('Torques τ_x(—) τ_y(--) τ_z(···)', fontweight='bold', fontsize=11)
    ax_tau.legend(fontsize=7); ax_tau.grid(True, alpha=0.3)
    ax_tau.set_ylim(-sc.MAX_TORQUE * 1.15, sc.MAX_TORQUE * 1.15)

    # ── Statistics text block ──
    def _pct(threshold_m):
        n_ok = sum(1 for ep in episodes
                   if ep['pos_error_m'][-1] < threshold_m)
        return f'{n_ok}/{num_starts} ({n_ok/num_starts*100:.0f}%)'

    mean_errs  = [np.mean(ep['pos_error_m'])  for ep in episodes]
    final_errs = [ep['pos_error_m'][-1]       for ep in episodes]
    # convergence time: first time error < 1 m and stays there for 2 s
    dt = sc.DT
    conv_ts = []
    for ep in episodes:
        mask = ep['pos_error_m'] < 1.0
        ct   = ep['time'][-1]
        for k in range(len(mask) - int(2/dt)):
            if np.all(mask[k:k + int(2/dt)]):
                ct = ep['time'][k]; break
        conv_ts.append(ct)

    term_counts = {}
    for ep in episodes:
        t = ep['termination']
        term_counts[t] = term_counts.get(t, 0) + 1

    lines = [
        'PERFORMANCE STATISTICS',
        '─' * 36,
        f'Episodes       : {num_starts}',
        f'Start radius   : 1 – {start_radius} m',
        '',
        'Mean pos error (whole traj):',
        f'  Avg  {np.mean(mean_errs)*1000:6.0f} mm',
        f'  Best {np.min(mean_errs)*1000:6.0f} mm',
        f'  Worst{np.max(mean_errs)*1000:6.0f} mm',
        '',
        'Final pos error:',
        f'  Avg  {np.mean(final_errs)*1000:6.0f} mm',
        f'  Best {np.min(final_errs)*1000:6.0f} mm',
        f'  Worst{np.max(final_errs)*1000:6.0f} mm',
        '',
        'Final success rate:',
        f'  < 500 mm : {_pct(0.5)}',
        f'  < 1000 mm: {_pct(1.0)}',
        f'  < 2000 mm: {_pct(2.0)}',
        '',
        'Convergence (<1 m, 2 s):',
        f'  Mean : {np.mean(conv_ts):.1f} s',
        f'  Min  : {np.min(conv_ts):.1f} s',
        '',
        'Terminations:',
    ] + [f'  {k}: {v}' for k, v in sorted(term_counts.items())]

    fig.text(0.975, 0.5, '\n'.join(lines),
             fontsize=8.5, family='monospace', va='center', ha='right',
             bbox=dict(boxstyle='round', fc='#f5f5f5', ec='#888888', alpha=0.85))

    fig.suptitle(
        f'Integrated DDPG – Multi-Start Analysis  ({num_starts} Starts, radius ≤ {start_radius} m)',
        fontsize=14, fontweight='bold')

    out = os.path.join(save_dir, f'multi_start_{num_starts}starts.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\n✓ Saved → {out}')

    # ── Console summary ──
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'  Mean pos error  : {np.mean(mean_errs)*1000:.0f} mm')
    print(f'  Final success   : {_pct(1.0)} under 1 m')
    print(f'  Mean conv. time : {np.mean(conv_ts):.1f} s')
    for k, v in sorted(term_counts.items()):
        print(f'  {k}: {v}')
    print('='*70)

    return episodes


# ──────────────────────────────────────────────────────────────────────────────
#  3.  Single-episode control detail plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_control_detail(env, agent, start_pos=None, save_path='control_detail.png',
                        wind_std=None):
    """
    Detailed single-episode breakdown:
      Row 1 : x/y/z position tracking
      Row 2 : u/v/w velocity
      Row 3 : phi/theta/psi attitude
      Row 4 : Ft + tau_x/y/z raw controls
      Row 5 : reward components
    """
    sc = env.sys_cfg
    print('\nRunning control-detail episode...')
    ep = run_episode(env, agent, start_pos=start_pos, wind_std=wind_std)
    t  = ep['time']

    fig, axes = plt.subplots(5, 1, figsize=(16, 20), facecolor='white',
                             sharex=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93,
                        bottom=0.05, hspace=0.38)

    labels_pos = ['x', 'y', 'z']
    labels_vel = ['u', 'v', 'w']
    labels_att = ['φ (roll)', 'θ (pitch)', 'ψ (yaw)']
    colors_xyz = ['#e05c5c', '#5cb85c', '#5b9bd5']

    # Row 0 – position
    ax = axes[0]
    for j, (lbl, c) in enumerate(zip(labels_pos, colors_xyz)):
        ax.plot(t, ep['pos_actual'][:, j],  color=c,  lw=2,   label=f'{lbl} actual')
        ax.plot(t, ep['pos_desired'][:, j], color=c,  lw=1.5,
                ls='--', alpha=0.6, label=f'{lbl} desired')
    ax.set_ylabel('Position (m)'); ax.set_title('Position Tracking', fontweight='bold')
    ax.legend(ncol=3, fontsize=8); ax.grid(True, alpha=0.3)

    # Row 1 – velocity
    ax = axes[1]
    for j, (lbl, c) in enumerate(zip(labels_vel, colors_xyz)):
        ax.plot(t, ep['vel_actual'][:, j],  color=c, lw=2,   label=f'{lbl} actual')
        ax.plot(t, ep['vel_desired'][:, j], color=c, lw=1.5,
                ls='--', alpha=0.6, label=f'{lbl} desired')
    ax.set_ylabel('Velocity (m/s)'); ax.set_title('Velocity Tracking', fontweight='bold')
    ax.legend(ncol=3, fontsize=8); ax.grid(True, alpha=0.3)

    # Row 2 – attitude
    ax = axes[2]
    for j, (lbl, c) in enumerate(zip(labels_att, colors_xyz)):
        ax.plot(t, ep['attitude_deg'][:, j], color=c, lw=2, label=lbl)
    ax.axhline(0, color='k', lw=0.8, alpha=0.4)
    lim = np.rad2deg(sc.CRASH_ANGLE) * 0.6
    ax.axhline( lim, color='red', ls=':', lw=1.5, alpha=0.5, label=f'±{lim:.0f}° soft limit')
    ax.axhline(-lim, color='red', ls=':', lw=1.5, alpha=0.5)
    ax.set_ylabel('Angle (°)'); ax.set_title('Attitude', fontweight='bold')
    ax.legend(ncol=4, fontsize=8); ax.grid(True, alpha=0.3)

    # Row 3 – controls
    ax = axes[3]
    # Dual y-axis: thrust on left, torques on right
    ax2 = ax.twinx()
    l1, = ax.plot(t, ep['Ft'], color='navy', lw=2, label=f'Ft (N)')
    ax.axhline(sc.MASS * sc.GRAVITY, color='navy', ls='--', lw=1.2, alpha=0.5)
    ax.axhline(sc.MAX_THRUST,        color='navy', ls=':',  lw=1,   alpha=0.4)
    ax.set_ylim(0, sc.MAX_THRUST * 1.08)
    ax.set_ylabel('Thrust Ft (N)', color='navy')
    ax.tick_params(axis='y', labelcolor='navy')

    tau_colors = ['#cc3333', '#33aa33', '#3333cc']
    tau_labels = ['τ_x', 'τ_y', 'τ_z']
    ls_list    = ['-', '--', ':']
    tau_lines  = []
    for j, (lbl, c, ls) in enumerate(zip(tau_labels, tau_colors, ls_list)):
        l, = ax2.plot(t, ep['tau'][:, j], color=c, lw=1.8, ls=ls, label=lbl)
        tau_lines.append(l)
    ax2.axhline( sc.MAX_TORQUE,     color='red', ls=':', lw=0.8, alpha=0.3)
    ax2.axhline(-sc.MAX_TORQUE,     color='red', ls=':', lw=0.8, alpha=0.3)
    ax2.set_ylim(-sc.MAX_TORQUE * 1.15, sc.MAX_TORQUE * 1.15)
    ax2.set_ylabel('Torque (Nm)', color='#883333')
    ax2.tick_params(axis='y', labelcolor='#883333')
    ax.set_title('Controls', fontweight='bold')
    combined = [l1] + tau_lines
    ax.legend(combined, [l.get_label() for l in combined], ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 4 – reward components
    ax = axes[4]
    rc = ep['reward_components']
    comp_colors = plt.cm.Set2(np.linspace(0, 1, len(rc)))
    for (k, v), c in zip(rc.items(), comp_colors):
        ax.plot(t, v, lw=1.8, color=c, label=k, alpha=0.85)
    ax.plot(t, ep['reward'], 'k-', lw=2.5, alpha=0.7, label='total')
    ax.axhline(0, color='k', lw=0.8, alpha=0.4)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Reward')
    ax.set_title('Reward Components', fontweight='bold')
    ax.legend(ncol=4, fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Integrated DDPG – Control Detail  '
        f'(start={np.round(ep["start_pos"],2)}  '
        f'term={ep["termination"]}  '
        f'mean_err={np.mean(ep["pos_error_mm"]):.0f} mm)',
        fontsize=13, fontweight='bold')

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'✓ Saved → {save_path}')
    return ep


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Trajectory Visualization – Integrated DDPG Agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Agent
    parser.add_argument('--agent', type=str, default=r'results\run_20260301_101610\integrated_phase_4_ep1000.pth',
                        help='Path to integrated agent .pth checkpoint')

    # Modes
    parser.add_argument('--video',       action='store_true',
                        help='Create animated GIF/MP4')
    parser.add_argument('--multi_start', action='store_true',
                        help='Create multi-start comparison PNG')
    parser.add_argument('--control',     action='store_true',
                        help='Create single-episode control detail PNG')
    parser.add_argument('--all',         action='store_true',
                        help='Run all three modes')

    # Shared
    parser.add_argument('--num_starts',   type=int,   default=8)
    parser.add_argument('--start_radius', type=float, default=4.0)
    parser.add_argument('--duration',     type=float, default=50.0,
                        help='Episode duration for animation (s)')
    parser.add_argument('--output_dir',   type=str,
                        default='results/visualization')
    parser.add_argument('--wind',         action='store_true',
                        help='Enable wind in evaluation')

    # Video-specific
    parser.add_argument('--video_starts', type=int,   default=3)
    parser.add_argument('--fps',          type=int,   default=20)
    parser.add_argument('--video_name',   type=str,   default='trajectory.gif')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('\n' + '='*70)
    print('TRAJECTORY VISUALIZATION  –  INTEGRATED DDPG AGENT')
    print('='*70)

    # ── Build environment ──
    int_cfg = IntegratedAgentConfig()
    sc      = SystemConfig()
    env     = IntegratedEnv(config=int_cfg)
    env.set_trajectory_scale(1.0)
    wind_std = [0.0003, 0.0003, 0.00015] if args.wind else [0, 0, 0]
    env.set_wind(wind_std)

    # ── Load agent ──
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
    if args.agent and os.path.exists(args.agent):
        agent.load(args.agent)
        print(f'\n✓ Loaded agent: {args.agent}')
    elif args.agent:
        print(f'\n⚠  Checkpoint not found: {args.agent}')
        print('   Using randomly initialised weights (for testing layout only)')
    else:
        print('\n⚠  No --agent specified. Using random weights.')

    # ── Run requested modes ──
    if args.all or args.multi_start:
        plot_multi_start_comparison(
            env, agent,
            num_starts   = args.num_starts,
            start_radius = args.start_radius,
            save_dir     = args.output_dir)

    if args.all or args.control:
        plot_control_detail(
            env, agent,
            wind_std  = wind_std if args.wind else None,
            save_path = os.path.join(args.output_dir, 'control_detail.png'))

    if args.all or args.video:
        create_animated_video(
            env, agent,
            save_path    = os.path.join(args.output_dir, args.video_name),
            num_starts   = args.video_starts,
            start_radius = args.start_radius,
            fps          = args.fps,
            duration     = args.duration)

    if not (args.all or args.multi_start or args.control or args.video):
        parser.print_help()
        print('\nNo mode selected. Add --multi_start, --video, --control, or --all.')

    print(f'\n✓ Done. Output → {args.output_dir}')


if __name__ == '__main__':
    main()
