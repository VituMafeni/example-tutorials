# Module 06 — Deep Q-Network on CartPole (PyTorch)

A reinforcement learning agent trained from scratch: a small neural
network that learns, purely from trial and reward, to balance a pole on a
moving cart. Implements a Deep Q-Network (replay buffer, target network,
epsilon-greedy exploration) directly — no RL library does the learning for
you.

- **Kernel type:** PyTorch (GPU)
- **Environment:** `CartPole-v1` from Gymnasium — no dataset download, the
  environment is simulated locally
- **Extra packages:** `gymnasium` — installed by the first cell/step
  (`pip install -q gymnasium`)
- **Est. GPU time:** ~10 minutes for 400 episodes

## Step-by-step: JupyterLab

1. Spawn or switch your JupyterLab session to a **PyTorch (GPU)** kernel
   (see the [top-level README](../README.md) if you haven't cloned this
   repo yet).
2. In a JupyterLab terminal:
   ```bash
   cd ~/ml-platform/docs/demos/student-examples/06-rl-dqn-cartpole-pytorch
   ```
3. Open `notebook.ipynb` from the file browser, confirm the kernel shown
   top-right is a **PyTorch** kernel, then `Run` → `Run All Cells`.

## Step-by-step: VS Code or Terminal

This folder also has `script.py` — the identical walkthrough as
`notebook.ipynb`, runnable via the platform's browser VS Code
("VS Code ↗") or browser Terminal ("Terminal ↗"):

```bash
cd ~/ml-platform/docs/demos/student-examples/06-rl-dqn-cartpole-pytorch
python script.py
```

Plots save as PNG files under `results/` instead of displaying inline.
**Read the storage warning in the [top-level README's VS Code/Terminal
section](../README.md#step-by-step-vs-code-or-terminal) first** — work
here does not survive a kernel stop/restart unless you click
"Sync to Home."

## What you'll learn

- The agent/environment/reward loop that underlies reinforcement learning
- Why RL needs a replay buffer and a separate target network to train
  stably
- Epsilon-greedy exploration and epsilon decay
- Reading a noisy reward curve (moving average) to judge whether an agent
  is actually learning

The material ends with four numbered exercises (exploration decay rate,
Double DQN, soft target updates, and a harder environment) plus an empty
cell/section for you to try them in.

> Note: this model is tiny (a 2-layer MLP), so GPU utilization will look
> low in `nvidia-smi` — that's expected. The point of this module is the
> RL training loop, not GPU-bound compute.
