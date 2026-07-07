"""
E1 + E2 combined run (they share the same simulated episodes).

E1 - Performance Recovery: completion rate, crash rate, episode length vs mu,
     compared across Oracle / phi_rma / Blind controllers.

E2 - Risk Calibration: per-mu rho (friction-circle utilization) distributions
     and cross-grip rho VARIANCE per controller. A calibrated controller
     should maintain similar rho across grip levels (adapting speed/steering
     to stay at a consistent risk level); an uncalibrated one will show high
     rho variance across mu (e.g., driving at the same aggression regardless
     of available grip).

Since both experiments read the exact same trajectories, we run the sim
ONCE and save full rho traces, enabling both analyses without re-simulating.
"""
import yaml, json, argparse
import numpy as np
import torch
from f1tenth_research.envs import F1TenthRMAEnv
from f1tenth_research.models import RMAActorCritic, AdaptationModule
from f1tenth_research.eval.safety_eval_harness import SafetyEvalController, compute_rho


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase1_checkpoint', required=True)
    parser.add_argument('--phase2_checkpoint', required=True)
    parser.add_argument('--track', default='sepang_tight')
    parser.add_argument('--history_window', type=int, default=10)
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=3000)
    parser.add_argument('--mu_sweep', type=float, nargs='+', default=[1.0, 0.8, 0.6, 0.4])
    parser.add_argument('--out', default='/research_ws/e1_e2_results.json')
    args = parser.parse_args()

    with open('f1tenth_research/configs/rma_config.yaml') as f:
        config = yaml.safe_load(f)

    env = F1TenthRMAEnv(config=config, track=args.track, max_episode_steps=args.max_steps)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ac = RMAActorCritic(obs_dim=obs_dim, env_params_dim=7, action_dim=action_dim)
    ckpt = torch.load(args.phase1_checkpoint, map_location='cpu', weights_only=False)
    ac.load_state_dict(ckpt['actor_critic'])
    ac.eval()

    adaptation = AdaptationModule(state_action_dim=obs_dim+action_dim,
                                   history_window=args.history_window, intrinsics_dim=8)
    adapt_ckpt = torch.load(args.phase2_checkpoint, map_location='cpu', weights_only=False)
    adaptation.load_state_dict(adapt_ckpt['adaptation'])
    adaptation.eval()

    controller = SafetyEvalController(ac, adaptation, history_window=args.history_window,
                                       obs_dim=obs_dim, action_dim=action_dim)

    results = {}  # results[mu][mode] = {episode summaries + per-episode rho lists}

    for mu in args.mu_sweep:
        results[str(mu)] = {}
        for mode in ['oracle', 'phi_rma', 'blind']:
            episode_summaries = []
            all_rho_this_condition = []

            for ep in range(args.n_episodes):
                obs, info = env.reset()
                env.current_physics_params['grip_factor'] = mu
                controller.reset_history()

                rho_trace = []
                prev_v = 0.0
                done, step, crashed = False, 0, False

                while not done and step < args.max_steps:
                    physics = dict(env.current_physics_params)
                    action, z = controller.act(obs, mode, physics)
                    obs, r, term, trunc, info = env.step(action)
                    env.current_physics_params['grip_factor'] = mu

                    done = term or trunc
                    if term:
                        crashed = True
                        # Skip recording rho for the crash-causing step -- obs here
                        # reflects post-collision state, contaminating the
                        # calibration measure with crash dynamics rather than
                        # genuine steady-state driving behavior.
                    else:
                        v = float(obs[0])
                        yaw_rate = float(obs[4])
                        rho = compute_rho(v, yaw_rate, prev_v, dt=0.02, mu=mu)
                        rho_trace.append(rho)
                        prev_v = v
                    step += 1


                episode_summaries.append({
                    'length': step,
                    'crashed': crashed,
                    'avg_rho': float(np.mean(rho_trace)) if rho_trace else 0.0,
                    'rho_trace': rho_trace,  # kept for E2's variance analysis
                })
                all_rho_this_condition.extend(rho_trace)

            lengths = [e['length'] for e in episode_summaries]
            crash_count = sum(1 for e in episode_summaries if e['crashed'])

            # Split rho stats by crashed vs survived to check if calibration
            # signal is being driven by crash dynamics
            crashed_rhos = [e['avg_rho'] for e in episode_summaries if e['crashed'] and e['rho_trace']]
            survived_rhos = [e['avg_rho'] for e in episode_summaries if not e['crashed'] and e['rho_trace']]

            results[str(mu)][mode] = {
                # E1 metrics
                'avg_length': float(np.mean(lengths)),
                'std_length': float(np.std(lengths)),
                'crash_rate': crash_count / args.n_episodes,
                'completion_rate': sum(1 for l in lengths if l >= args.max_steps*0.9) / args.n_episodes,
                # E2 metrics
                'mean_rho': float(np.mean(all_rho_this_condition)) if all_rho_this_condition else 0.0,
                'std_rho': float(np.std(all_rho_this_condition)) if all_rho_this_condition else 0.0,
                'mean_rho_crashed_eps': float(np.mean(crashed_rhos)) if crashed_rhos else None,
                'mean_rho_survived_eps': float(np.mean(survived_rhos)) if survived_rhos else None,
                'episodes': episode_summaries,
            }
            print(f"mu={mu} mode={mode}: avg_len={results[str(mu)][mode]['avg_length']:.0f} "
                  f"crash_rate={results[str(mu)][mode]['crash_rate']:.2f} "
                  f"mean_rho={results[str(mu)][mode]['mean_rho']:.3f}")

    # E2 headline: cross-grip rho VARIANCE per controller (the calibration metric)
    print("\n=== E2 Calibration Summary: cross-grip mean_rho variance per controller ===")
    for mode in ['oracle', 'phi_rma', 'blind']:
        mode_rhos_by_mu = [results[str(mu)][mode]['mean_rho'] for mu in args.mu_sweep]
        variance = float(np.var(mode_rhos_by_mu))
        print(f"{mode}: rho across mu levels = {[f'{r:.3f}' for r in mode_rhos_by_mu]}, "
              f"variance = {variance:.5f}")

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
