"""
DDPG Agent – Integrated Controller
=====================================
Deep Deterministic Policy Gradient (Lillicrap et al., 2015) adapted for
the integrated single-agent quadcopter controller.

How DDPG differs from TD3
--------------------------
  TD3 (what was here before)          DDPG (this file)
  ─────────────────────────────────   ──────────────────────────────────
  Twin Q-networks (Q1 + Q2)           Single Q-network
  Clipped double-Q target             Standard Bellman target  Q(s,a)
  Target policy smoothing             No smoothing noise on target policy
  Delayed actor update (every K)      Actor updated every step
  Gaussian exploration noise          Ornstein-Uhlenbeck process noise
                                        (temporally correlated, better for
                                        continuous physical control)

Architecture (identical to TD3 version for fair comparison)
------------------------------------------------------------
  Actor  : input → LayerNorm+ReLU → 3× residual blocks → tanh × max_action
  Critic : (state ‖ action) → 3-layer MLP → scalar Q-value
  Both use soft target networks (Polyak averaging, τ = 0.005)

Action convention
-----------------
  Actor output ∈ [-1, +1]  (max_action = 1.0)
  Physical rescaling done in IntegratedEnv.step():
      a[0] → Ft    ∈ [MIN_THRUST, MAX_THRUST]
      a[1] → tau_x ∈ [-MAX_TORQUE,     +MAX_TORQUE    ]
      a[2] → tau_y ∈ [-MAX_TORQUE,     +MAX_TORQUE    ]
      a[3] → tau_z ∈ [-MAX_TORQUE_YAW, +MAX_TORQUE_YAW]
"""

# this file is not fully provided to prevent original research.