"""Evaluate a trained agent on Drone Hunter."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Register the environment
import drone_hunter.envs  # noqa: F401

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def evaluate(
    model_path: Path | str,
    vec_normalize_path: Path | str | None = None,
    n_episodes: int = 5,
    render: bool = True,
    fps: int = 30,
    grid_size: int = 8,
    deterministic: bool = True,
    single_target_mode: bool = False,
) -> dict:
    """Evaluate a trained model.

    Args:
        model_path: Path to the trained model (.zip)
        vec_normalize_path: Path to VecNormalize stats (.pkl)
        n_episodes: Number of episodes to run
        render: Whether to display the game
        fps: Frames per second for display
        grid_size: Firing grid size
        deterministic: Use deterministic actions
        single_target_mode: Use single-target observation mode

    Returns:
        Dictionary with evaluation stats
    """
    model_path = Path(model_path)

    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Create environment
    def make_env():
        return gym.make(
            "DroneHunter-v0",
            render_mode="rgb_array" if render else None,
            grid_size=grid_size,
            max_frames=1000,
            oracle_mode=True,
            single_target_mode=single_target_mode,
        )

    env = DummyVecEnv([make_env])

    # Load normalization stats if available
    if vec_normalize_path and Path(vec_normalize_path).exists():
        print(f"Loading VecNormalize from: {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False

    # Setup display
    if render and HAS_OPENCV:
        cv2.namedWindow("Drone Hunter - Trained Agent", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drone Hunter - Trained Agent", 640, 640)

    # Run episodes
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    episode_hits = []
    episode_misses = []

    frame_delay = int(1000 / fps)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\n--- Episode {ep + 1}/{n_episodes} ---")

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1

            if render and HAS_OPENCV:
                # Get frame from underlying env
                frame = env.envs[0].render()
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Drone Hunter - Trained Agent", frame_bgr)
                    key = cv2.waitKey(frame_delay)
                    if key == ord('q'):
                        done = True
                        break

        # Get final info
        final_info = info[0]
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_scores.append(final_info.get('score', 0))
        episode_hits.append(final_info.get('hits', 0))
        episode_misses.append(final_info.get('misses', 0))

        print(f"Reward: {total_reward:.2f} | Steps: {steps} | "
              f"Score: {final_info.get('score', 0)} | "
              f"Hits: {final_info.get('hits', 0)} | "
              f"Misses: {final_info.get('misses', 0)} | "
              f"Reason: {final_info.get('game_over_reason', 'N/A')}")

    # Cleanup
    if render and HAS_OPENCV:
        cv2.destroyAllWindows()
    env.close()

    # Summary stats
    import numpy as np

    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_score": np.mean(episode_scores),
        "mean_hits": np.mean(episode_hits),
        "mean_misses": np.mean(episode_misses),
        "hit_rate": np.sum(episode_hits) / max(1, np.sum(episode_hits) + np.sum(episode_misses)),
    }

    print(f"\n{'=' * 50}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Episodes: {n_episodes}")
    print(f"Mean Reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    print(f"Mean Episode Length: {stats['mean_length']:.1f}")
    print(f"Mean Score: {stats['mean_score']:.1f}")
    print(f"Mean Hits: {stats['mean_hits']:.1f}")
    print(f"Mean Misses: {stats['mean_misses']:.1f}")
    print(f"Hit Rate: {stats['hit_rate']:.1%}")
    print(f"{'=' * 50}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Drone Hunter agent")
    parser.add_argument("model_path", type=str, help="Path to model .zip file")
    parser.add_argument("--vec-normalize", type=str, default=None,
                       help="Path to VecNormalize .pkl file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--fps", type=int, default=30, help="Render FPS")
    parser.add_argument("--grid-size", type=int, default=8, help="Grid size")
    parser.add_argument("--stochastic", action="store_true",
                       help="Use stochastic actions instead of deterministic")
    parser.add_argument("--single-target", action="store_true",
                       help="Use single-target observation mode")

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        vec_normalize_path=args.vec_normalize,
        n_episodes=args.episodes,
        render=not args.no_render,
        fps=args.fps,
        grid_size=args.grid_size,
        deterministic=not args.stochastic,
        single_target_mode=args.single_target,
    )


if __name__ == "__main__":
    main()
