#!/usr/bin/env python3
"""
train_ppo.py — PPO Training Harness for NietzscheDB L-System Evolution Strategy.

AlphaZero-inspired self-play: the agent learns to optimise graph topology by
choosing one of the four GrowthActions defined in nietzsche-rl/src/env.rs:

    0: Balanced      (growth = pruning)
    1: FavorGrowth   (growth > pruning)
    2: FavorPruning  (pruning > growth)
    3: Consolidate   (structural repair / move toward centroid)

State (state_dim=64):
    Krylov subspace projection of graph health metrics. The first 5 features
    map directly to the HealthReport fields used by the Reactor:
      [0] mean_energy          (float, from HealthReport.mean_energy)
      [1] is_fractal           (0.0 or 1.0, from HealthReport.is_fractal)
      [2] gap_ratio            (gap_count / 80, clamped to [0,1])
      [3] entropy_spike_feat   (entropy_spike_count / 10, clamped to [0,1])
      [4] coherence_score      (float, from HealthReport.coherence_score)
    Features [5..63] are reserved for learned latent projections.

Reward:
    +1.0  if Hausdorff dimension preserved AND query-speed proxy improves
    -1.0  if Hausdorff dimension degrades (structure broken)
     0.0  neutral (no improvement, no degradation)

ONNX output name: ppo_growth_v1 (matches PpoConfig default and AGENCY_PPO_MODEL).
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os

# ---------------------------------------------------------------------------
# Constants — must match nietzsche-rl expectations
# ---------------------------------------------------------------------------
STATE_DIM = 64       # Krylov subspace projection (same as ValueNetwork)
NUM_ACTIONS = 4      # GrowthAction enum variants (Balanced/FavorGrowth/FavorPruning/Consolidate)
HIDDEN_DIM = 128     # Shared trunk width


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for L-System growth strategy selection.

    Architecture mirrors the ValueNetwork style (LayerNorm + GELU) but adds
    a categorical actor head for discrete action selection.
    """

    def __init__(self, state_dim=STATE_DIM, num_actions=NUM_ACTIONS, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Actor: action logits (no softmax — Categorical handles it)
        self.actor = nn.Linear(hidden_dim, num_actions)

        # Critic: scalar state-value
        self.critic = nn.Linear(hidden_dim, 1)

        # Xavier init for stable training start (same convention as ValueNetwork)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        h = self.shared(state)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value

    def act(self, state):
        """Sample an action and return (action, log_prob, value)."""
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value


# ---------------------------------------------------------------------------
# Environment (simulated graph health for self-play)
# ---------------------------------------------------------------------------
class GraphHealthEnv:
    """
    Simulated graph environment for self-play training.

    Generates synthetic graph health state vectors and computes rewards
    based on Hausdorff dimension preservation and query-speed proxy.
    """

    def __init__(self, state_dim=STATE_DIM, max_steps=200):
        self.state_dim = state_dim
        self.max_steps = max_steps
        self.state = None
        self.hausdorff_baseline = None
        self.step_count = 0
        self.reset()

    def reset(self):
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        # Normalise to Poincare ball (norm < 1.0)
        norm = np.linalg.norm(self.state)
        if norm > 0.99:
            self.state = self.state / norm * 0.99

        # Simulated Hausdorff dimension from first feature (mean_energy proxy)
        self.hausdorff_baseline = abs(self.state[0]) * 3.0 + 1.0
        self.step_count = 0
        return self.state.copy()

    def step(self, action: int):
        """Apply action, return (next_state, reward, done)."""
        self.step_count += 1

        # Action-specific perturbation to graph topology
        if action == 0:    # Balanced
            delta = np.random.randn(self.state_dim) * 0.01
        elif action == 1:  # FavorGrowth — increase complexity
            delta = np.random.randn(self.state_dim) * 0.03
            delta[0] += 0.02
        elif action == 2:  # FavorPruning — decrease complexity
            delta = np.random.randn(self.state_dim) * 0.02
            delta[0] -= 0.01
        else:              # Consolidate — drift toward centroid (stability)
            delta = -self.state * 0.05

        self.state = self.state + delta.astype(np.float32)

        # Re-enforce Poincare constraint
        norm = np.linalg.norm(self.state)
        if norm > 0.99:
            self.state = self.state / norm * 0.99

        # --- Reward ---
        current_hausdorff = abs(self.state[0]) * 3.0 + 1.0
        hausdorff_preserved = abs(current_hausdorff - self.hausdorff_baseline) < 0.5

        # Query-speed proxy: lower entropy across features [1:10] = faster lookups
        entropy_proxy = np.mean(np.abs(self.state[1:10]))
        speed_improved = entropy_proxy < 0.5

        if hausdorff_preserved and speed_improved:
            reward = 1.0
        elif not hausdorff_preserved:
            reward = -1.0
        else:
            reward = 0.0

        done = self.step_count >= self.max_steps
        return self.state.copy(), reward, done


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalised Advantage Estimation (Schulman et al., 2016)."""
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages


# ---------------------------------------------------------------------------
# ONNX export helper
# ---------------------------------------------------------------------------
class PolicyOnly(nn.Module):
    """Extracts only the actor head for lean ONNX inference in nietzsche-rl."""

    def __init__(self, full_model):
        super().__init__()
        self.shared = full_model.shared
        self.actor = full_model.actor

    def forward(self, x):
        return torch.softmax(self.actor(self.shared(x)), dim=-1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    epochs=50,
    steps_per_epoch=2048,
    mini_batch_size=64,
    ppo_epochs=10,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    lr=3e-4,
    max_grad_norm=0.5,
    data_path="../../checkpoints/health_trajectories.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PPO] Training on {device}")

    policy = PPOActorCritic().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Attempt to load real trajectory data; fall back to simulated env
    real_data = None
    if os.path.exists(data_path):
        print(f"[PPO] Loading real trajectory data from {data_path}")
        real_data = torch.load(data_path)
    else:
        print(f"[PPO] No real data at {data_path}. Using simulated graph environment.")

    env = GraphHealthEnv()

    best_reward = float("-inf")

    for epoch in range(epochs):
        # -- Collect trajectories --
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        state = env.reset()
        episode_reward = 0.0
        episode_rewards = []

        for _ in range(steps_per_epoch):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                action, log_prob, value = policy.act(state_t)

            next_state, reward, done = env.step(action.item())

            states.append(state)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(float(done))
            values.append(value.item())

            episode_reward += reward
            state = next_state

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                state = env.reset()

        # -- GAE & returns --
        advantages = compute_gae(rewards, values, dones)
        returns = [a + v for a, v in zip(advantages, values)]

        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        old_log_probs_t = torch.FloatTensor(log_probs).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)
        returns_t = torch.FloatTensor(returns).to(device)

        # Normalise advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # -- PPO clipped update --
        for _ in range(ppo_epochs):
            indices = torch.randperm(steps_per_epoch)
            for start in range(0, steps_per_epoch, mini_batch_size):
                end = start + mini_batch_size
                idx = indices[start:end]

                logits, value = policy(states_t[idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[idx])
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_t[idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(value.squeeze(), returns_t[idx])
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        print(f"Epoch [{epoch + 1}/{epochs}] - Avg Reward: {avg_reward:.2f} | Episodes: {len(episode_rewards)}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            os.makedirs("../../checkpoints", exist_ok=True)
            torch.save(policy.state_dict(), "../../checkpoints/ppo_growth_v1.pt")
            print(f"  -> New best model saved! Reward: {best_reward:.2f}")

    # ------------------------------------------------------------------
    # Export actor head to ONNX (this is what PpoEngine loads at runtime)
    # ------------------------------------------------------------------
    print("\n[PPO] Exporting actor to ONNX...")
    policy.eval()
    dummy_input = torch.randn(1, STATE_DIM).to(device)

    policy_only = PolicyOnly(policy).to(device)
    policy_only.eval()

    with torch.no_grad():
        sample_probs = policy_only(dummy_input)
        print(f"[PPO] Sample action probs: {sample_probs.cpu().numpy().round(4)}")

    os.makedirs("../../models", exist_ok=True)
    onnx_path = "../../models/ppo_growth_v1.onnx"

    torch.onnx.export(
        policy_only,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["action_probs"],
        dynamic_axes={
            "state":        {0: "batch_size"},
            "action_probs": {0: "batch_size"},
        },
    )
    print(f"[PPO] Model exported -> {onnx_path}")

    # Quick validation
    try:
        import onnx
        m = onnx.load(onnx_path)
        onnx.checker.check_model(m)
        print("[PPO] ONNX model validation passed.")
    except ImportError:
        print("[PPO] onnx package not installed -- skipping validation.")

    print(f"[PPO] Training complete. Best avg reward: {best_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training for NietzscheDB L-System evolution")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=2048, help="Rollout steps per epoch")
    parser.add_argument("--mini-batch-size", type=int, default=64, help="Mini-batch size for PPO updates")
    parser.add_argument("--ppo-epochs", type=int, default=10, help="SGD passes per epoch")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--data-path", default="../../checkpoints/health_trajectories.pt",
                        help="Path to real trajectory data (optional)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        lr=args.lr,
        data_path=args.data_path,
    )
