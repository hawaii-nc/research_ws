"""
E1 - Performance Recovery
Sweep grip (mu) over several levels. For each level, run N episodes per
controller (Oracle / phi_rma / Blind). Report completion rate, crash rate,
and episode length vs mu for each controller.

Pass criterion (RMA-standard): phi_rma recovers performance close to Oracle,
clearly better than Blind, under grip shift.
"""
import yaml, json, argparse
import numpy as np
import torch
from f1tenth_research.envs import F1TenthRMAEnv
from f1tenth_research.models import RMAActorCritic, AdaptationModule
from f1tenth_research.eval.safety_eval_harness import SafetyEvalController, run_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase1_checkpoint', required=True)
    parser.add_argument('--phase2_checkpoint', required=True)
    parser.add_argument('--track', default='sepang_tight')
    parser.add_argument('--history_window', type=int, default=10)
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=3000)
    parser.add_argument('--mu_sweep', type=float, nargs='+', default=[1.0, 0.8, 0.6, 0.4])
    parser.add_argument('--out', default='/research_ws/e1_results.json')
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

    results = {}
    for mu in args.mu_sweep:
        # Force fixed grip for this sweep point by monkey-patching the randomizer's sample
        env.randomizer.config['enabled'] = True
        results[mu] = {}
        for mode in ['oracle', 'phi_rma', 'blind']:
            lengths, crashes = [], 0
            for ep in range(args.n_episodes):
                obs, info = env.reset()
                # Force this episode's grip_factor to the target mu
                env.current_physics_params['grip_factor'] = mu
                controller.reset_history()
                done, step = False, 0
                while not done and step < args.max_steps:
                    physics = env.current_physics_params
                    action, z = controller.act(obs, mode, physics)
                    obs, r, term, trunc, info = env.step(action)
                    env.current_physics_params['grip_factor'] = mu  # hold fixed all episode
                    done = term or trunc
                    step += 1
                lengths.append(step)
                if term:
                    crashes += 1
            results[mu][mode] = {
                'avg_length': float(np.mean(lengths)),
                'std_length': float(np.std(lengths)),
                'crash_rate': crashes / args.n_episodes,
                'completion_rate': sum(1 for l in lengths if l >= args.max_steps*0.9) / args.n_episodes,
            }
            print(f"mu={mu} mode={mode}: avg_len={results[mu][mode]['avg_length']:.0f} "
                  f"crash_rate={results[mu][mode]['crash_rate']:.2f}")

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
