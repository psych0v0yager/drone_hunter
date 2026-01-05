"""Training script for Drone Hunter using PPO."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Register the environment
import drone_hunter.envs  # noqa: F401


def make_env(
    grid_size: int = 8,
    max_drones: int = 10,
    max_frames: int = 1000,
    oracle_mode: bool = True,
    single_target_mode: bool = False,
    detection_noise: float = 0.02,
    detection_dropout: float = 0.05,
) -> gym.Env:
    """Create a Drone Hunter environment.

    Args:
        grid_size: Size of the firing grid.
        max_drones: Maximum tracked drones per frame.
        max_frames: Maximum frames per episode.
        oracle_mode: Use ground truth positions.
        single_target_mode: If True, only pass most urgent drone.
        detection_noise: Noise std for simulated detector (detector mode only).
        detection_dropout: Probability of missing detection (detector mode only).

    Returns:
        Gymnasium environment.
    """
    return gym.make(
        "DroneHunter-v0",
        grid_size=grid_size,
        max_drones=max_drones,
        max_frames=max_frames,
        oracle_mode=oracle_mode,
        single_target_mode=single_target_mode,
        detection_noise=detection_noise,
        detection_dropout=detection_dropout,
    )


def train(
    total_timesteps: int = 1_000_000,
    grid_size: int = 8,
    max_frames: int = 1000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    net_arch: list[int] | None = None,
    output_dir: Path | str = "runs",
    run_name: str | None = None,
    checkpoint_freq: int = 50_000,
    eval_freq: int = 10_000,
    verbose: int = 1,
    single_target_mode: bool = False,
    norm_obs: bool = True,
    oracle_mode: bool = True,
    detection_noise: float = 0.02,
    detection_dropout: float = 0.05,
) -> PPO:
    """Train a PPO agent on Drone Hunter.

    Args:
        total_timesteps: Total training timesteps.
        grid_size: Size of the firing grid.
        max_frames: Maximum frames per episode.
        n_envs: Number of parallel environments.
        learning_rate: Learning rate.
        n_steps: Steps per update.
        batch_size: Minibatch size.
        n_epochs: Epochs per update.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_range: PPO clip range.
        ent_coef: Entropy coefficient.
        output_dir: Directory for outputs.
        run_name: Name for this run.
        checkpoint_freq: Checkpoint save frequency.
        eval_freq: Evaluation frequency.
        verbose: Verbosity level.
        single_target_mode: If True, use simplified single-target observation.
        norm_obs: If True, normalize observations with VecNormalize.
        oracle_mode: If True, use ground truth. If False, use Kalman tracker.
        detection_noise: Noise std for simulated detector (detector mode only).
        detection_dropout: Probability of missing detection (detector mode only).

    Returns:
        Trained PPO model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"drone_hunter_{timestamp}"

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training run: {run_name}")
    print(f"Output directory: {run_dir}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Single-target mode: {single_target_mode}")
    print(f"Oracle mode: {oracle_mode}")
    if not oracle_mode:
        print(f"Detection noise: {detection_noise}")
        print(f"Detection dropout: {detection_dropout}")
    print(f"Observation normalization: {norm_obs}")
    print("-" * 50)

    # Create vectorized environment
    def env_fn():
        return make_env(
            grid_size=grid_size,
            max_frames=max_frames,
            oracle_mode=oracle_mode,
            single_target_mode=single_target_mode,
            detection_noise=detection_noise,
            detection_dropout=detection_dropout,
        )

    vec_env = make_vec_env(env_fn, n_envs=n_envs)

    # Optionally normalize observations
    vec_env = VecNormalize(
        vec_env,
        norm_obs=norm_obs,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # Create evaluation environment
    eval_env = make_vec_env(env_fn, n_envs=1)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=norm_obs,
        norm_reward=False,  # Don't normalize eval rewards
        clip_obs=10.0,
        training=False,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_drone_hunter",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Configure policy network architecture
    if net_arch is None:
        net_arch = [256, 256]  # Larger default than SB3's 64x64

    import torch.nn as nn
    policy_kwargs = {
        "net_arch": dict(pi=net_arch, vf=net_arch),
        "activation_fn": nn.ReLU,  # ReLU instead of Tanh
    }

    print(f"Policy network: {net_arch} with ReLU activation")

    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=verbose,
        tensorboard_log=str(run_dir / "tensorboard"),
        policy_kwargs=policy_kwargs,
    )

    print(f"Model policy: {model.policy}")
    print("-" * 50)

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = run_dir / "final_model"
    model.save(str(final_path / "model"))
    vec_env.save(str(final_path / "vec_normalize.pkl"))

    print("-" * 50)
    print(f"Training complete!")
    print(f"Final model saved to: {final_path}")

    return model


def main() -> None:
    """Entry point for training."""
    parser = argparse.ArgumentParser(description="Train Drone Hunter agent with PPO")

    # Environment
    parser.add_argument("--grid-size", type=int, default=8, help="Firing grid size")
    parser.add_argument("--max-frames", type=int, default=1000, help="Max frames per episode")

    # Training
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments")

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256],
                       help="Policy network architecture (e.g., 256 256 for 2 layers)")

    # Output
    parser.add_argument("--output-dir", type=str, default="runs", help="Output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency")

    # Misc
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity (0-2)")

    # Single-target mode (simplified observation)
    parser.add_argument("--single-target", action="store_true",
                       help="Use single-target mode (most urgent drone only)")
    parser.add_argument("--no-norm-obs", action="store_true",
                       help="Disable observation normalization (preserve one-hot encoding)")

    # Detector mode (Kalman tracker instead of oracle)
    parser.add_argument("--detector-mode", action="store_true",
                       help="Use detector mode with Kalman tracker (no ground truth)")
    parser.add_argument("--detection-noise", type=float, default=0.02,
                       help="Noise std for simulated detector (detector mode only)")
    parser.add_argument("--detection-dropout", type=float, default=0.05,
                       help="Probability of missing detection (detector mode only)")

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        grid_size=args.grid_size,
        max_frames=args.max_frames,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        net_arch=args.net_arch,
        output_dir=args.output_dir,
        run_name=args.run_name,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        verbose=args.verbose,
        single_target_mode=args.single_target,
        norm_obs=not args.no_norm_obs,
        oracle_mode=not args.detector_mode,
        detection_noise=args.detection_noise,
        detection_dropout=args.detection_dropout,
    )


if __name__ == "__main__":
    main()
