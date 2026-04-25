"""
agent.py — PPO agent (actor-critic) for CartPole-v1

Beginner-friendly overview (advanced idea, simple words):

- PPO (Proximal Policy Optimization: a policy-gradient method that updates the policy but *prevents*
  it from changing too much at once) learns a policy π(a|s) directly.

- Actor-Critic (two-headed model):
  - Actor (policy network): outputs action probabilities πθ(a|s)
  - Critic (value network): outputs Vφ(s) (expected future return from state s)

- GAE (Generalized Advantage Estimation: a way to compute "advantage" signals that are less noisy):
  Advantage A(s,a) roughly answers: "was this action better or worse than what the critic expected?"

This file contains:
- A small neural net with shared backbone and two heads (actor + critic)
- Utilities to compute log-probabilities, entropy bonus, value estimates
- GAE computation and PPO clipped loss update
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOHyperParams:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 256


class ActorCritic(nn.Module):
    """
    Shared backbone → actor logits (2 actions) and critic value (1).

    For CartPole:
    - state_dim = 4
    - action_dim = 2 (discrete)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, action_dim)  # outputs logits (unnormalized scores)
        self.critic = nn.Linear(hidden, 1)  # outputs V(s)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: np.ndarray, device: torch.device) -> tuple[int, float, float]:
        """
        Sample an action from π(a|s) and return:
        - action (int)
        - log_prob (float): log π(a|s)
        - value (float): V(s)
        """
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray, device: torch.device) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, _ = self.forward(x)
        return int(torch.argmax(logits, dim=-1).item())


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GAE (Generalized Advantage Estimation) computes advantages using TD residuals.

    Terms:
    - TD error / delta δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    - Advantage A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...

    Inputs are length T (rollout length):
    - rewards[t], values[t], dones[t] (done mask: 1 if episode ended at step t)
    - last_value: V(s_T) used for bootstrapping at the end of rollout if not done
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=torch.float32, device=rewards.device)

    gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, hp: PPOHyperParams, device: torch.device) -> None:
        self.hp = hp
        self.device = device
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=hp.lr)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        """
        PPO update (multiple epochs over the same rollout data).

        Core PPO idea:
        - We want to maximize: E[ ratio * advantage ]
          where ratio = π_new(a|s) / π_old(a|s)
        - But we CLIP the ratio into [1-ε, 1+ε] so the policy cannot change too much at once.
          (This is the "proximal" part.)
        """
        hp = self.hp

        # Normalize advantages (standard PPO trick to reduce scale issues)
        adv = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        n = states.shape[0]
        idxs = np.arange(n)

        last_stats: dict[str, float] = {}
        for _ in range(hp.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, hp.minibatch_size):
                mb = idxs[start : start + hp.minibatch_size]

                logits, values = self.model(states[mb])
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_log_probs[mb])
                unclipped = ratio * adv[mb]
                clipped = torch.clamp(ratio, 1.0 - hp.clip_ratio, 1.0 + hp.clip_ratio) * adv[mb]
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = F.mse_loss(values, returns[mb])

                loss = policy_loss + hp.value_coef * value_loss - hp.entropy_coef * entropy

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), hp.max_grad_norm)
                self.optim.step()

                approx_kl = (old_log_probs[mb] - log_probs).mean().item()
                clip_frac = (torch.abs(ratio - 1.0) > hp.clip_ratio).float().mean().item()

                last_stats = {
                    "loss_total": float(loss.item()),
                    "loss_policy": float(policy_loss.item()),
                    "loss_value": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "approx_kl": float(approx_kl),
                    "clip_frac": float(clip_frac),
                    "adv_mean": float(advantages.mean().item()),
                }

        return last_stats

