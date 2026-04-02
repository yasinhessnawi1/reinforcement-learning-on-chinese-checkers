"""
train_ppo.py — Train a MaskablePPO agent on ChineseCheckersEnv.

Usage
-----
python src/training/train_ppo.py --opponent greedy --num-envs 8 --total-timesteps 500000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'multi system single machine minimal'))

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.agents.greedy_agent import greedy_policy
from src.agents.advanced_heuristic import advanced_heuristic_policy
from src.network.feature_extractor import ResNetFeaturesExtractor
from src.training.self_play import CheckpointPool, make_self_play_opponent


def mask_fn(env: ChineseCheckersEnv):
    """Return the action mask for the current env state."""
    return env.action_masks()


def make_env(opponent_policy=None, max_steps: int = 200, rank: int = 0, seed: int = 0):
    """
    Factory that returns a callable creating a single wrapped env instance.

    Parameters
    ----------
    opponent_policy : callable or None
        Policy function for the opponent. None = random.
    max_steps : int
        Max steps per episode.
    rank : int
        Index of this env in the vec pool (used to offset seeds).
    seed : int
        Base random seed.

    Returns
    -------
    Callable[[], gym.Env]
    """
    def _init():
        env = ChineseCheckersEnv(opponent_policy=opponent_policy, max_steps=max_steps)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(opponent_policy, num_envs: int, max_steps: int, seed: int = 0):
    """Create a vectorised env pool (SubprocVecEnv for >1 env, DummyVecEnv for 1)."""
    env_fns = [make_env(opponent_policy=opponent_policy, max_steps=max_steps, rank=i, seed=seed)
               for i in range(num_envs)]

    if num_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


class SelfPlayPoolCallback(BaseCallback):
    """Saves the current model to the checkpoint pool every N steps."""

    def __init__(self, pool: CheckpointPool, save_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.pool = pool
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.pool.save(self.model, self.num_timesteps)
            if self.verbose:
                print(f"[SelfPlay] Saved checkpoint to pool at step {self.num_timesteps} "
                      f"(pool size: {self.pool.size()})")
        return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO on ChineseCheckersEnv."
    )

    # Environment
    parser.add_argument("--opponent", choices=["none", "random", "greedy", "advanced", "self"], default="greedy",
                        help="Opponent policy: 'none' (solo), 'random', 'greedy', 'advanced', or 'self' (default: greedy)")
    parser.add_argument("--self-play-pool-dir", type=str, default=None,
                        help="Directory for self-play checkpoint pool (default: <model_dir>/pool)")
    parser.add_argument("--self-play-save-freq", type=int, default=500_000,
                        help="Steps between saving model to self-play pool (default: 500000)")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Number of parallel envs (default: 8)")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode (default: 200)")

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per rollout per env (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size (default: 64)")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Epochs per update (default: 10)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda (default: 0.95)")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO clip range (default: 0.2)")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="Entropy coefficient (default: 0.0)")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value function coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Max gradient norm (default: 0.5)")

    # Logging / saving
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for saving (default: auto-generated)")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000,
                        help="Checkpoint save frequency in steps (default: 50000)")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluation frequency in steps (default: 10000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")

    # Network
    parser.add_argument("--policy", choices=["mlp", "resnet", "vit", "masked-vit"], default="resnet",
                        help="Policy type: 'mlp', 'resnet', 'vit', or 'masked-vit' (default: resnet)")
    parser.add_argument("--num-blocks", type=int, default=6,
                        help="ResNet blocks (default: 6, only used with --policy resnet)")
    parser.add_argument("--num-filters", type=int, default=64,
                        help="ResNet filters (default: 64, only used with --policy resnet)")
    parser.add_argument("--vit-d-model", type=int, default=128,
                        help="Transformer width (default: 128, only used with --policy vit/masked-vit)")
    parser.add_argument("--vit-n-heads", type=int, default=4,
                        help="Transformer attention heads (default: 4)")
    parser.add_argument("--vit-n-layers", type=int, default=4,
                        help="Transformer layers (default: 4)")
    parser.add_argument("--vit-features-dim", type=int, default=256,
                        help="Transformer output features dim (default: 256)")

    # Resume / fine-tune
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved model (.zip) to resume training or fine-tune from")

    # Visualization
    parser.add_argument("--viz-freq", type=int, default=10_000,
                        help="Steps between eval visualizations. 0 = disabled (default: 10000)")

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Auto-generate run name if not provided
    if args.run_name is None:
        args.run_name = f"ppo_{args.opponent}_envs{args.num_envs}_steps{args.total_timesteps}"

    model_dir = os.path.join("models", args.run_name)
    log_dir = os.path.join("experiments", args.run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Run name    : {args.run_name}")
    print(f"Opponent    : {args.opponent}")
    print(f"Num envs    : {args.num_envs}")
    print(f"Total steps : {args.total_timesteps:,}")
    print(f"Model dir   : {model_dir}")
    print(f"Log dir     : {log_dir}")
    if args.resume:
        print(f"Resuming from: {args.resume}")

    # --- Resolve opponent policy ---
    if args.opponent == "none":
        opponent_policy = "none"  # solo mode — no opponent moves
    elif args.opponent == "greedy":
        opponent_policy = greedy_policy
    elif args.opponent == "advanced":
        opponent_policy = advanced_heuristic_policy
    elif args.opponent == "self":
        pool_dir = args.self_play_pool_dir or os.path.join(model_dir, "pool")
        pool = CheckpointPool(pool_dir, max_size=20)
        # Seed pool with resume checkpoint if provided
        if args.resume and pool.size() == 0:
            import shutil
            seed_path = os.path.join(pool_dir, "checkpoint_0.zip")
            shutil.copy(args.resume, seed_path)
            print(f"[SelfPlay] Seeded pool with {args.resume}")
        opponent_policy = make_self_play_opponent(pool, greedy_fallback_ratio=0.3)
        print(f"[SelfPlay] Pool dir: {pool_dir}, save freq: {args.self_play_save_freq:,}")
    else:
        opponent_policy = None  # random

    # --- Vectorised training env ---
    vec_env = build_vec_env(
        opponent_policy=opponent_policy,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    # --- Eval env (single, DummyVecEnv) — solo when training solo, else vs greedy ---
    eval_opponent = "none" if args.opponent == "none" else (greedy_policy if args.opponent in ("greedy", "advanced", "self") else None)
    eval_env = DummyVecEnv([
        make_env(opponent_policy=eval_opponent, max_steps=args.max_steps, rank=99, seed=args.seed)
    ])

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_interval // args.num_envs, 1),
        save_path=model_dir,
        name_prefix="checkpoint",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=log_dir,
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    callbacks = [checkpoint_cb, eval_cb]

    if args.viz_freq > 0:
        from src.visualization.viz_callback import VizCallback
        from src.visualization.replay_gui import start_viz_thread
        viz_queue = start_viz_thread()
        viz_opponent = "none" if args.opponent == "none" else eval_opponent
        callbacks.append(VizCallback(viz_queue, eval_freq=args.viz_freq, opponent_policy=viz_opponent))

    if args.opponent == "self":
        pool_cb = SelfPlayPoolCallback(
            pool=pool,
            save_freq=max(args.self_play_save_freq // args.num_envs, 1),
            verbose=1,
        )
        callbacks.append(pool_cb)

    # --- Model ---
    if args.resume:
        # Load existing model and update env + hyperparams for fine-tuning
        model = MaskablePPO.load(
            args.resume,
            env=vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=log_dir,
            verbose=1,
            seed=args.seed,
        )
    else:
        # --- Policy config ---
        if args.policy == "resnet":
            policy_str = "CnnPolicy"
            policy_kwargs = dict(
                features_extractor_class=ResNetFeaturesExtractor,
                features_extractor_kwargs=dict(
                    num_blocks=args.num_blocks,
                    num_filters=args.num_filters,
                ),
                share_features_extractor=True,
                net_arch=dict(pi=[256], vf=[256]),
            )
        elif args.policy in ("vit", "masked-vit"):
            from src.network.transformer_extractor import StandardViT, MaskedViT
            vit_class = StandardViT if args.policy == "vit" else MaskedViT
            policy_str = "MlpPolicy"  # MlpPolicy accepts flat features — avoids triple-extractor bug
            policy_kwargs = dict(
                features_extractor_class=vit_class,
                features_extractor_kwargs=dict(
                    d_model=args.vit_d_model,
                    n_heads=args.vit_n_heads,
                    n_layers=args.vit_n_layers,
                    features_dim=args.vit_features_dim,
                ),
                share_features_extractor=True,
                net_arch=dict(pi=[256], vf=[256]),
            )
        else:
            policy_str = "MlpPolicy"
            policy_kwargs = {}

        model = MaskablePPO(
            policy=policy_str,
            env=vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=log_dir,
            verbose=1,
            seed=args.seed,
        )

    print(f"\nPolicy architecture:\n{model.policy}\n")

    # --- Train ---
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # --- Save final model ---
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
