"""
Module 06 - Deep Q-Network on CartPole (PyTorch) - VS Code / plain-script version.

This is the exact same walkthrough as notebook.ipynb in this folder, saved
as a plain .py file so it also works in the platform's browser-based VS
Code (code-server) or in a terminal, without needing a notebook viewer.

Two ways to run it:
  1. VS Code (or any editor with the Python extension's cell support):
     each "# %%" marker below is a runnable cell - open this file and use
     "Run Cell" / "Run Below" the same way you would in a notebook.
  2. Plain terminal, top to bottom:
       python script.py
     Plots are saved as PNG files under ./results/ (in addition to trying
     to display inline) so they're visible either way.

IMPORTANT - storage: code in VS Code / a terminal on this platform runs on
the kernel's own EPHEMERAL disk, not your persistent JupyterLab home
folder. Click "Sync to Home" in the portal's My Workloads panel BEFORE you
stop, restart, or walk away from this kernel, or your work here is lost.
See the top-level student-examples/README.md, "Step-by-step: VS Code or
Terminal" section.
"""


# %% [markdown]
# # Module 06 — Deep Q-Network on CartPole (PyTorch)
#
# **Goal:** train a reinforcement learning agent from scratch — a small
# neural network that learns, purely from trial and reward, to balance a
# pole on a moving cart.
#
# **Before you start:** launch with a **PyTorch (GPU)** kernel (see this
# module's [README](README.md)).

# %%
import torch

assert torch.cuda.is_available(), (
    "No GPU detected in this kernel.\n"
    "Go back to the portal's Servers tab, stop this session, spawn a new one "
    "using the kernel named in this module's README, then run this again."
)
device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))
print("Total GPU memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))

# %% [markdown]
# ## 0. Install the environment library

# %%
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gymnasium"])

# %% [markdown]
# ## 1. Look at the environment

# %%
import gymnasium as gym

env = gym.make("CartPole-v1")
state, info = env.reset(seed=0)
print("State (cart position, cart velocity, pole angle, pole angular velocity):", state)
print("Action space:", env.action_space, "(0 = push left, 1 = push right)")

# %% [markdown]
# ## 2. Define the Q-network and replay buffer

# %%
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.tensor(state, dtype=torch.float32),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(next_state, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

policy_net = QNetwork().to(device)
target_net = QNetwork().to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()

# %% [markdown]
# ## 3. Train with epsilon-greedy exploration

# %%
import numpy as np

num_episodes = 400
batch_size = 64
gamma = 0.99
epsilon, epsilon_min, epsilon_decay = 1.0, 0.02, 0.995
target_update_every = 10

episode_rewards = []

for episode in range(1, num_episodes + 1):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                action = q_values.argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        total_reward += reward

        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            states, actions = states.to(device), actions.to(device)
            rewards, next_states, dones = rewards.to(device), next_states.to(device), dones.to(device)

            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next_q = target_net(next_states).max(dim=1).values
                target = rewards + gamma * max_next_q * (1 - dones)
            loss = F.mse_loss(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

    if episode % target_update_every == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 20 == 0:
        avg_recent = np.mean(episode_rewards[-20:])
        print(f"episode {episode}/{num_episodes}  avg_reward(last 20)={avg_recent:.1f}  epsilon={epsilon:.3f}")

# %% [markdown]
# ## 4. Plot the learning curve

# %%
import matplotlib.pyplot as plt

window = 20
moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")

plt.plot(episode_rewards, alpha=0.3, label="episode reward")
plt.plot(range(window - 1, len(episode_rewards)), moving_avg, label=f"{window}-episode moving average")
plt.axhline(475, color="green", linestyle="--", label="considered 'solved'")
plt.xlabel("episode")
plt.ylabel("total reward")
plt.legend()
plt.title("CartPole DQN learning curve")
import os
os.makedirs("results", exist_ok=True)
plt.savefig("results/plot_01.png", dpi=120, bbox_inches="tight")
print("Saved plot to results/plot_01.png")
plt.show()

# %% [markdown]
# ## 5. Watch the trained agent (greedy policy)

# %%
eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
frames = []
state, _ = eval_env.reset(seed=1)
done = False
total_reward = 0

while not done:
    frames.append(eval_env.render())
    with torch.no_grad():
        action = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)).argmax(dim=1).item()
    state, reward, terminated, truncated, _ = eval_env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"Greedy evaluation episode reward: {total_reward}")

fig, axes = plt.subplots(1, 6, figsize=(16, 3))
step_indices = np.linspace(0, len(frames) - 1, 6, dtype=int)
for ax, idx in zip(axes, step_indices):
    ax.imshow(frames[idx])
    ax.set_title(f"step {idx}")
    ax.axis("off")
plt.tight_layout()
import os
os.makedirs("results", exist_ok=True)
plt.savefig("results/plot_02.png", dpi=120, bbox_inches="tight")
print("Saved plot to results/plot_02.png")
plt.show()

# %% [markdown]
# ## Exercises
#
# 1. **Faster/slower exploration decay.** Try `epsilon_decay = 0.99` vs
#    `0.999` — how does that change how many episodes it takes to solve the
#    task?
# 2. **Double DQN.** Change the target computation to use `policy_net` to
#    pick the action and `target_net` to evaluate it (this reduces Q-value
#    overestimation) — does the learning curve get smoother?
# 3. **Soft target updates.** Instead of copying weights every 10 episodes,
#    blend them every step: `target = tau * policy + (1 - tau) * target`
#    with `tau = 0.005`.
# 4. **Stretch goal:** try a harder environment, e.g.
#    `gym.make("LunarLander-v3")` (8-dim state, 4 actions) — you will need
#    more episodes and a bigger network.

# %%
# Your experiments here
