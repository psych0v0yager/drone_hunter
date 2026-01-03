"""Drone Hunter - RL environment for drone detection training.

Usage:
    # Play as human
    uv run python -m drone_hunter.scripts.play

    # Train agent
    uv run python -m drone_hunter.scripts.train

    # Quick demo
    uv run python main.py
"""

import gymnasium as gym

# Register the environment
import drone_hunter.envs  # noqa: F401


def main():
    """Quick demo of the Drone Hunter environment."""
    print("Drone Hunter - Quick Demo")
    print("=" * 40)

    # Create environment
    env = gym.make("DroneHunter-v0", grid_size=8, max_frames=100)

    obs, info = env.reset()
    print(f"Observation shape: detections={obs['detections'].shape}, "
          f"game_state={obs['game_state'].shape}")
    print(f"Action space: {env.action_space}")

    # Run a few random steps
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"\nEpisode ended at step {step}: {info['game_over_reason']}")
            break

    print(f"\nFinal score: {info['score']}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Hits: {info['hits']} | Misses: {info['misses']}")

    env.close()
    print("\nTo play as human: uv run python -m drone_hunter.scripts.play")
    print("To train agent:   uv run python -m drone_hunter.scripts.train")


if __name__ == "__main__":
    main()
