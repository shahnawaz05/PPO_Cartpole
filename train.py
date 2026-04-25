"""
train.py — PPO training loop for CartPole-v1

What happens in this file (end-to-end):

1) Create the Gymnasium environment (CartPole-v1).
2) Repeatedly collect a rollout of N steps using the current policy:
   - For each step, store: state, action, reward, done, log_prob_old, value_old
3) At the end of the rollout, compute:
   - GAE advantages A_t
   - Returns R_t (targets for the critic)
4) Run PPO updates for multiple epochs on that same rollout data (minibatches):
   - Actor loss: clipped policy gradient objective
   - Critic loss: MSE(value, return)
   - Entropy bonus: encourages exploration
5) Log episode rewards + save:
   - results/episode_rewards.json
   - results/learning_curve_ppo.png
   - results/vs_dqn_comparison.png (only if DQN metrics exist)
   - results/trained_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import PPOAgent, PPOHyperParams, compute_gae


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def moving_average(x: list[float], window: int) -> list[float]:
    if len(x) < window:
        return []
    arr = np.asarray(x, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    out = np.convolve(arr, kernel, mode="valid")
    return out.tolist()


def maybe_plot_vs_dqn(ppo_rewards: list[float]) -> None:
    """
    If the DQN project metrics exist, plot PPO vs DQN returns.

    We compare episode return vs episode index (simple and readable).
    If you want sample-efficiency vs steps, you can extend this later.
    """
    dqn_metrics = ROOT.parent / "Deep QNetwork_Cartpole" / "mini_project_1_dqn_cartpole" / "results" / "training_metrics.json"
    if not dqn_metrics.is_file():
        return

    try:
        dqn = json.loads(dqn_metrics.read_text(encoding="utf-8"))
        dqn_rewards = dqn.get("episode_rewards", [])
        if not dqn_rewards:
            return
    except Exception:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(dqn_rewards, alpha=0.35, label="DQN (episode return)")
    plt.plot(ppo_rewards, alpha=0.35, label="PPO (episode return)")

    dqn_ma = moving_average(list(map(float, dqn_rewards)), window=20)
    ppo_ma = moving_average(list(map(float, ppo_rewards)), window=20)
    if dqn_ma:
        plt.plot(range(19, 19 + len(dqn_ma)), dqn_ma, linewidth=2.0, label="DQN MA(20)")
    if ppo_ma:
        plt.plot(range(19, 19 + len(ppo_ma)), ppo_ma, linewidth=2.0, label="PPO MA(20)")

    plt.axhline(500, color="r", linestyle="--", label="Max return (500)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO vs DQN on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "vs_dqn_comparison.png", dpi=150)
    plt.close()


def train(
    total_timesteps: int = 200_000,
    n_steps: int = 2048,
    seed: int = 42,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)

    hp = PPOHyperParams(
        # matches your spec defaults
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        update_epochs=10,
        minibatch_size=256,
    )
    agent = PPOAgent(state_dim, action_dim, hp, device=device)

    # Rollout buffers (size n_steps)
    states = torch.zeros((n_steps, state_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((n_steps,), dtype=torch.int64, device=device)
    rewards = torch.zeros((n_steps,), dtype=torch.float32, device=device)
    dones = torch.zeros((n_steps,), dtype=torch.float32, device=device)
    old_log_probs = torch.zeros((n_steps,), dtype=torch.float32, device=device)
    values = torch.zeros((n_steps,), dtype=torch.float32, device=device)

    episode_returns: list[float] = []
    episode_return = 0.0

    obs, _ = env.reset()
    global_step = 0
    update_idx = 0

    while global_step < total_timesteps:
        # -------------------------
        # 1) Collect rollout
        # -------------------------
        for t in range(n_steps):
            action, logp, v = agent.model.act(obs, device=device)

            states[t] = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions[t] = int(action)
            old_log_probs[t] = float(logp)
            values[t] = float(v)

            next_obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rewards[t] = float(r)
            dones[t] = 1.0 if done else 0.0

            episode_return += float(r)
            global_step += 1

            obs = next_obs

            if done:
                episode_returns.append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset()

            if global_step >= total_timesteps:
                # if we hit budget early, shrink effective rollout length
                effective_steps = t + 1
                break
        else:
            effective_steps = n_steps

        # Bootstrap value for last state (if rollout ended mid-episode)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_v = agent.model(x)
            last_value = last_v.squeeze(0)

        adv, rets = compute_gae(
            rewards=rewards[:effective_steps],
            values=values[:effective_steps],
            dones=dones[:effective_steps],
            last_value=last_value,
            gamma=hp.gamma,
            gae_lambda=hp.gae_lambda,
        )

        # -------------------------
        # 2) PPO update
        # -------------------------
        stats = agent.update(
            states=states[:effective_steps],
            actions=actions[:effective_steps],
            old_log_probs=old_log_probs[:effective_steps],
            advantages=adv,
            returns=rets,
        )

        update_idx += 1
        if update_idx % 10 == 0 and episode_returns:
            tail = episode_returns[-10:]
            print(
                f"Update {update_idx} | steps={global_step} | "
                f"mean_return(last10)={float(np.mean(tail)):.1f} | "
                f"policy_loss={stats.get('loss_policy', 0.0):.3f} | "
                f"kl~={stats.get('approx_kl', 0.0):.5f}"
            )

    env.close()

    # -------------------------
    # Save artifacts
    # -------------------------
    (RESULTS_DIR / "episode_rewards.json").write_text(
        json.dumps(
            {
                "episode_returns": episode_returns,
                "total_timesteps": total_timesteps,
                "n_steps": n_steps,
                "seed": seed,
                "hyperparams": hp.__dict__,
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    torch.save(
        {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "model_state_dict": agent.model.state_dict(),
            "hyperparams": hp.__dict__,
        },
        RESULTS_DIR / "trained_model.pt",
    )

    plt.figure(figsize=(10, 5))
    plt.plot(episode_returns, alpha=0.5, label="Episode return")
    ma = moving_average(episode_returns, window=20)
    if ma:
        plt.plot(range(19, 19 + len(ma)), ma, linewidth=2.0, label="Moving avg (20)")
    plt.axhline(500, color="r", linestyle="--", label="Max return (500)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve_ppo.png", dpi=150)
    plt.close()

    maybe_plot_vs_dqn(episode_returns)

    print(f"\nSaved: {RESULTS_DIR / 'trained_model.pt'}")
    print(f"Saved: {RESULTS_DIR / 'episode_rewards.json'}")
    print(f"Saved: {RESULTS_DIR / 'learning_curve_ppo.png'}")
    if (RESULTS_DIR / "vs_dqn_comparison.png").is_file():
        print(f"Saved: {RESULTS_DIR / 'vs_dqn_comparison.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train PPO on CartPole-v1")
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train(total_timesteps=args.total_timesteps, n_steps=args.n_steps, seed=args.seed)


if __name__ == "__main__":
    main()

