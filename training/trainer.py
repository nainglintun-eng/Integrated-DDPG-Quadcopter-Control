"""
Integrated Single-Agent Trainer  –  FIXED VERSION
===================================================
Fix: warmup random actions are in [-1, +1] (normalised) to match the
     fixed actor output space. Previously they were in [-MAX_THRUST, +MAX_THRUST]
     which gave physically valid thrust but wildly out-of-range torques,
     poisoning the replay buffer with unrepresentative transitions.
"""

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_integrated_controller(env, agent, config, save_dir):
    from configs.config import TrainingConfig
    tc = TrainingConfig()

    print('\n' + '='*70)
    print('INTEGRATED SINGLE-AGENT TD3 TRAINING  –  FIXED')
    print('='*70)
    print(f'  Phases      : {len(config.CURRICULUM_PHASES)}')
    print(f'  State dim   : {config.STATE_DIM}')
    print(f'  Action dim  : {config.ACTION_DIM}  (normalised [-1,+1], rescaled in env)')
    print(f'  max_action  : {config.ACTION_SCALE}  (actor tanh bound)')
    print(f'  Device      : {agent.device}')
    print('='*70 + '\n')

    all_stats = {k: [] for k in ['rewards','pos_errors','vel_errors',
                                  'successes','episode_lengths','q_values',
                                  'exploration_noise','phases']}

    for phase_idx, phase in enumerate(config.CURRICULUM_PHASES):
        print(f'\n{"="*70}')
        print(f'PHASE {phase_idx+1}/{len(config.CURRICULUM_PHASES)}: {phase["name"].upper()}')
        print(f'{"="*70}')
        print(f'  Trajectory scale : {phase["trajectory_scale"]*100:.0f}%')
        print(f'  Max steps        : {phase["max_steps"]} ({phase["max_steps"]*env.sys_cfg.DT:.1f}s)')
        print(f'  Success target   : < {phase["success_error"]*1000:.0f} mm')
        print(f'  Exploration      : {phase["exploration_start"]:.2f} → {phase["exploration_end"]:.2f}')
        print(f'{"="*70}\n')

        env.set_trajectory_scale(phase['trajectory_scale'])
        env.set_wind(phase.get('wind_std', [0,0,0]))
        env.set_start_radius(phase.get('start_radius', 0.0))
        env.max_steps = phase['max_steps']

        phase_stats = _run_phase(env, agent, phase, phase_idx+1, save_dir, tc)

        for key in ['rewards','pos_errors','vel_errors','successes',
                    'episode_lengths','q_values','exploration_noise']:
            all_stats[key].extend(phase_stats[key])
        all_stats['phases'].extend([phase_idx+1]*len(phase_stats['rewards']))

        phase_path = f'{save_dir}/integrated_phase_{phase_idx+1}.pth'
        agent.save(phase_path)
        print(f'\n✓ Phase {phase_idx+1} complete  →  {phase_path}\n')

    print('\n' + '='*70)
    print('TRAINING COMPLETE')
    print('='*70)
    w = min(100, len(all_stats['rewards']))
    print(f'  Episodes     : {len(all_stats["rewards"])}')
    print(f'  Success rate : {np.mean(all_stats["successes"][-w:]):.1%}')
    print(f'  Pos error    : {np.mean(all_stats["pos_errors"][-w:]):.1f} mm')
    print('='*70 + '\n')
    return all_stats


def _run_phase(env, agent, phase_config, phase_num, save_dir, tc):
    stats = {k: [] for k in ['rewards','pos_errors','vel_errors','successes',
                              'episode_lengths','q_values','exploration_noise']}

    total_eps       = phase_config['episodes']
    expl_start      = phase_config.get('exploration_start', 0.3)
    expl_end        = phase_config.get('exploration_end',   0.05)
    consecutive_ok  = 0
    best_mean_error = float('inf')

    for episode in range(total_eps):
        progress      = episode / max(total_eps - 1, 1)
        current_noise = expl_start + (expl_end - expl_start) * progress
        agent.set_noise_scale(current_noise)

        state     = env.reset()
        ep_reward = 0.0
        ep_pos    = []
        ep_vel    = []
        ep_q      = []
        ep_len    = 0
        done      = False

        # Reset OU noise at the start of each episode (DDPG).
        # For TD3 / Gaussian noise this is a no-op.
        if hasattr(agent, 'reset_noise'):
            agent.reset_noise()

        while not done:
            if agent.total_steps < agent.warmup_steps:
                # Random actions in normalised [-1, +1] space
                action = np.random.uniform(-1.0, 1.0, agent.action_dim)
            else:
                action = agent.select_action(state, explore=True)

            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.append((state, action, reward, next_state, done))

            if agent.total_steps >= agent.warmup_steps:
                losses = agent.update(batch_size=agent.batch_size)
                if losses.get('q_value') is not None:
                    ep_q.append(losses['q_value'])

            ep_reward += reward
            ep_pos.append(info['pos_error'])
            ep_vel.append(info['vel_error'])
            ep_len    += 1
            agent.total_steps += 1
            state = next_state

        agent.episode_count += 1

        mean_pos = np.mean(ep_pos)
        mean_vel = np.mean(ep_vel)
        mean_q   = np.mean(ep_q) if ep_q else 0.0

        success = (mean_pos < phase_config['success_error'] * 1000 and
                   info.get('termination') == 'max_steps')
        consecutive_ok = consecutive_ok + 1 if success else 0

        stats['rewards'].append(ep_reward)
        stats['pos_errors'].append(mean_pos)
        stats['vel_errors'].append(mean_vel)
        stats['successes'].append(1 if success else 0)
        stats['episode_lengths'].append(ep_len)
        stats['q_values'].append(mean_q)
        stats['exploration_noise'].append(current_noise)

        w              = min(100, len(stats['rewards']))
        recent_reward  = np.mean(stats['rewards'][-w:])
        recent_error   = np.mean(stats['pos_errors'][-w:])
        recent_success = np.mean(stats['successes'][-w:])

        if (episode + 1) % tc.LOG_FREQUENCY == 0 or episode < 3:
            print(f'Phase {phase_num} | Ep {episode+1}/{total_eps} | '
                  f'R={ep_reward:7.2f} (avg {recent_reward:7.2f}) | '
                  f'Err={mean_pos:6.1f}mm (avg {recent_error:6.1f}mm) | '
                  f'len={ep_len} term={info.get("termination","?")} | '
                  f'ok={consecutive_ok} noise={current_noise:.3f} Q={mean_q:.3f}')

        if recent_error < best_mean_error:
            best_mean_error = recent_error
            if tc.KEEP_BEST_ONLY:
                agent.save(f'{save_dir}/integrated_phase_{phase_num}_best.pth')

        if (episode + 1) % tc.SAVE_FREQUENCY == 0:
            agent.save(f'{save_dir}/integrated_phase_{phase_num}_ep{episode+1}.pth')

        if (consecutive_ok >= phase_config['required_successes'] and
                episode >= tc.MIN_EPISODES_BEFORE_STOP):
            print(f'\n✓✓✓ Phase {phase_num} CONVERGED at episode {episode+1}!  '
                  f'err={mean_pos:.1f}mm\n')
            remaining = total_eps - episode - 1
            for _ in range(remaining):
                for k in stats: stats[k].append(stats[k][-1])
            break

    return stats
