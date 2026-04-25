"""
evaluate.py — evaluate a trained PPO policy on CartPole-v1

What happens here:
- Load the saved actor-critic weights from results/trained_model.pt
- Run several episodes with a *greedy* policy (choose argmax action)
- Print returns and mean return

Why greedy for eval?
- PPO is stochastic during training (samples actions).
- For reporting, greedy often shows the learned “best” behavior clearly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from agent import ActorCritic


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def evaluate(
    episodes: int = 10,
    model_path: Path | None = None,
    render: bool = False,
    seed: int = 123,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = model_path or (RESULTS_DIR / "trained_model.pt")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Train first: python train.py")

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])

    model = ActorCritic(state_dim, action_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    returns: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        total = 0.0

        while not (terminated or truncated):
            action = model.greedy_action(obs, device=device)
            obs, r, terminated, truncated, _ = env.step(action)
            total += float(r)

        returns.append(total)
        print(f"Eval episode {ep + 1}/{episodes} | return: {total:.0f}")

    env.close()
    print(f"\nMean return over {episodes} episodes: {float(np.mean(returns)):.1f} (max 500)")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate PPO on CartPole-v1")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--model", type=str, default="", help="Path to trained_model.pt (default: results/trained_model.pt)")
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    model_path = Path(args.model) if args.model else None
    evaluate(episodes=args.episodes, model_path=model_path, render=args.render)


if __name__ == "__main__":
    main()

