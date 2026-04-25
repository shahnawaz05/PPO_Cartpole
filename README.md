# PPO on CartPole (Mini Project 2)

This project implements **PPO (Proximal Policy Optimization: a modern policy-gradient algorithm that updates the policy while preventing overly-large changes)** on **CartPole-v1** (Gymnasium). It’s a strong portfolio RL baseline because it shows you can build a full **actor-critic (policy + value)** pipeline and explain **policy gradients**.

---

## Problem (CartPole)

You control a cart that can be pushed **left** or **right**. A pole is attached to the cart, and you want to keep it upright.

- **State (observation)**: `[cart_pos, cart_vel, pole_angle, pole_angular_vel]` (4 numbers)
- **Actions**: `{0: push left, 1: push right}` (discrete)
- **Reward**: +1 each timestep you keep balancing
- **Max return**: 500 (CartPole-v1 ends at 500 steps if perfect)

**Success** = evaluation mean return near **500**.

---

## Key insight: Policy Gradient vs Q-Learning

### DQN / Q-learning style (Project 1)
- Learn **Q(s,a) (action-value: expected future reward if you take action a in state s)**.
- Derive policy from Q: \(\pi(s)=\arg\max_a Q(s,a)\).
- **Off-policy (can learn from older data stored in replay buffer)**.

### PPO / Policy Gradient style (this project)
- Learn policy **directly**: \(\pi_\theta(a|s)\) (probability of action given state).
- Update parameters \(\theta\) to increase expected return.
- **On-policy (learn mainly from data collected with the current policy)**.

---

## How PPO works (beginner-friendly, but technical)

### Actor-Critic architecture

One neural network with two heads:

- **Actor (policy head)**: outputs **logits (unnormalized action scores)** → a **Categorical distribution** \(\pi(a|s)\)
- **Critic (value head)**: outputs **V(s) (state-value: expected future discounted return from state s)**

Network (as requested): **4 → 128 → ReLU → 128 → ReLU**, then:
- actor → 2 logits
- critic → 1 value

### Advantage and GAE

Policy gradient needs an **advantage (how much better an action was than expected)**:

\[
A_t = \hat{R}_t - V(s_t)
\]

But raw returns are noisy, so PPO often uses **GAE (Generalized Advantage Estimation: mixes multiple-step TD errors to reduce variance)**:

\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]
\[
A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots
\]

- **γ (gamma)**: discount factor (0.99)
- **λ (lambda)**: bias/variance tradeoff (0.95 is a common sweet spot)

### PPO clipped objective (the “proximal” part)

We compare the new policy to the old one using:

\[
ratio = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
\]

Then maximize a clipped objective:

\[
\max \; \mathbb{E}\left[\min(ratio \cdot A_t,\; \text{clip}(ratio, 1-\epsilon, 1+\epsilon)\cdot A_t)\right]
\]

This means:
- If the new policy tries to become **too different** from the old policy (ratio outside \([1-\epsilon,1+\epsilon]\)), the objective gets clipped.
- This prevents unstable “big jumps” that can break training.

---

## Project structure

```
2_PPO_CartPole/
├── train.py
├── agent.py
├── evaluate.py
├── requirements.txt
├── results/
│   ├── learning_curve_ppo.png
│   ├── vs_dqn_comparison.png        # created only if DQN metrics exist
│   ├── episode_rewards.json
│   └── trained_model.pt
└── README.md
```

---

## Troubleshooting (Windows / Python versions)

If `import torch` fails with **DLL load / initialization** errors, you’re likely on **very new Python (e.g. 3.14)** where **PyTorch** wheels may not work yet.

- Install **Python 3.10–3.12 (64-bit)** from [python.org](https://www.python.org/downloads/), create a new venv, then reinstall requirements.
- Or run the project on **Google Colab** and download the `results/` artifacts.

---

## How to run

From `c:\Users\sn4177\Desktop\Reinforcement Learning\2_PPO_CartPole`:

```bash
python -m venv .venv
# PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python evaluate.py
```

Useful flags:

```bash
python train.py --total-timesteps 200000 --n-steps 2048 --seed 42
python evaluate.py --episodes 10
python evaluate.py --render
```

---

## What output to expect

### During training (`train.py`)

Training prints every ~10 PPO updates (not every episode) because PPO trains from rollouts:

Example:
`Update 10 | steps=20480 | mean_return(last10)=... | policy_loss=... | kl~=...`

Interpretation:
- `mean_return(last10)` should trend upward toward **500** over time.
- `kl~` is an approximate **KL divergence (a measure of how much the policy distribution changed)**. PPO clipping tends to keep it small.

### After training

You should see these files:
- `results/trained_model.pt` (weights)
- `results/episode_rewards.json` (raw returns)
- `results/learning_curve_ppo.png` (plot)
- `results/vs_dqn_comparison.png` (only if your DQN metrics file exists)

### Evaluation (`evaluate.py`)

Prints per-episode returns and the mean. **Mean near 500** is success.

---

## How to measure success (what to report in your MS applications)

- **Greedy evaluation mean return**: aim **≥ 450–500** over 10 episodes.
- **Learning curve**: smooth increase and plateau near 500.
- **Sample efficiency (steps-to-solve)**: CartPole gives +1 per step, so total steps ≈ sum of episode returns in `episode_rewards.json`.

Gymnasium-style “solved” threshold often quoted: moving average return **≥ 475**.

---

## Results (fill after you run)

This run produced the following artifacts under `results/`:
- `episode_rewards.json`
- `learning_curve_ppo.png`

Summary from `results/episode_rewards.json`:

- **Total timesteps**: **200,000**  
- **Approx. environment steps** (sum of episode returns): **199,640**  
- **Best episode return**: **500 / 500**  
- **Mean return (last 10 episodes)**: **500.0 / 500**  
- **Mean return (last 50 episodes)**: **474.4 / 500**  
- **First time mean(last 10) ≥ 450**: **episode 947**  
- **First time mean(last 50) ≥ 450**: **episode 983**

Interpretation:
- The policy reaches the **maximum 500-step episode** many times.
- The moving average over the last 50 episodes is near the common “solved” threshold (**≥ 475**) for CartPole-v1.

## Plots

![PPO Learning Curve](results/learning_curve_ppo.png)

