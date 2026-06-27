# Integrated Trajectory Tracking with DDPG and TD3

Single-agent deep reinforcement learning controllers for quadcopter trajectory tracking. A single agent maps the full 18-dimensional observation directly to motor thrust and body torques, trained end-to-end on an expanding helical trajectory using either DDPG or TD3.

---

## Overview

Both agents share the same environment and network architecture, differing only in their learning algorithm:

| | DDPG | TD3 |
|---|---|---|
| Q-networks | Single | Twin (clipped double-Q) |
| Target policy | No smoothing | Gaussian smoothing noise |
| Actor update | Every step | Every 2 steps (delayed) |
| Exploration | Ornstein-Uhlenbeck noise | Gaussian noise |

The environment runs at **100 Hz** and uses **RK4 integration** for the 12-state quadcopter dynamics (`m=1 kg`, `L=0.2 m`, `Ixx/Iyy/Izz = 0.3/0.4/0.5 kg·m²`).

---

![Example Trajectory Tracking](examples/trajectory.gif)
![Example Trajectory Image](examples/final_trajectory.png)


## State and Action Space

**Observation (18-dim)**
```
[pos_error/MAX_POS (3), vel/MAX_VEL (3), att_rad (3),
 rates/10 (3), vel_des/MAX_VEL (3), acc_des/MAX_ACCEL (3)]
```

**Action (4-dim, actor outputs in `[-1, 1]`)**
```
a[0] → Ft    ∈ [0.981,  19.62]  N
a[1] → τx   ∈ [-1.962,  1.962] Nm
a[2] → τy   ∈ [-1.962,  1.962] Nm
a[3] → τz   ∈ [-1.373,  1.373] Nm
```
Each channel is rescaled independently inside `env.step()` to prevent torque saturation.

---

## Training

A four-phase curriculum progressively increases task difficulty:

| Phase | Trajectory | Max Steps | Episodes |
|---|---|---|---|
| 1 | Hover | 500 | 2 000 |
| 2 | Helix 25% | 1 250 | 2 000 |
| 3 | Helix 50% | 2 500 | 2 000 |
| 4 | Full helix | 5 000 | 4 000 |

![Training Log](examples/integrated_training.png)

The full trajectory is an expanding helix: `r(t) = A(1 - e^{-Bt})`, `ω = 0.2 rad/s`, `vz = 1 m/s`.

**Train DDPG:**
```bash
cd integrated_agent
python main.py
```


## Evaluation

```bash
# Full disturbance table (10 conditions)
python evaluate_performance_table.py \
    --integrated_td3  results/<run>/td3_final.pth \
    --integrated_ddpg results/<run>/ddpg_final.pth \
    --episodes 50

# MATLAB-faithful noise sweep (17 conditions: control noise, sensor noise, wind)
python evaluate_matlab_noise.py \
    --integrated_td3  results/<run>/td3_final.pth \
    --integrated_ddpg results/<run>/ddpg_final.pth \
    --episodes 30

# With motor-level saturation (saturate_controls)
python evaluate_matlab_noise.py --integrated_ddpg results/<run>/ddpg_final.pth \
    --saturate --episodes 30

# Quick demo without checkpoints
python evaluate_matlab_noise.py --demo
```

Outputs: LaTeX table (`.tex`), flat CSV, and NumPy dict (`.npy`) in `results_matlab/`.

---

## Key Hyperparameters

| Parameter | DDPG | TD3 |
|---|---|---|
| Hidden dims | 256 | 256 |
| Buffer size | 500 000 | 300 000 |
| Batch size | 512 | 256 |
| Actor LR | 5e-5 | 3e-5 |
| Critic LR | 3e-4 | 1e-4 |
| γ | 0.99 | 0.99 |
| τ (Polyak) | 0.005 | 0.005 |
| OU θ / σ | 0.15 / 0.4 | — |

---

## Project Structure

```
integrated_agent/
├── agents/
│   ├── ddpg_agent.py       # DDPG with OUNoise and residual actor
│   └── td3_agent.py        # TD3 with twin critics and delayed policy update
├── environments/
│   ├── integrated_env.py   # Gym env: obs, reward, per-channel action rescaling
│   └── dynamics.py         # RK4 quadcopter dynamics
├── configs/
│   └── config.py           # All hyperparameters and trajectory definition
├── training/
│   └── trainer.py          # Curriculum loop, early stopping, checkpoint saving
├── utils/
│   ├── evaluation.py       # Episode metrics (RMSE, convergence, crash rate)
│   └── visualization.py    # Training curves and 3-D trajectory plots
├── main.py                 # Entry point
└── trajectory_visualization.py

evaluate_performance_table.py   # 10-condition disturbance table
evaluate_matlab_noise.py        # 17-condition MATLAB-faithful noise sweep
```

---

## Requirements

```
python >= 3.12
gym==0.26.2
matplotlib==3.10.8
numpy==2.4.3
scipy==1.17.1
torch==2.6.0+cu126
```