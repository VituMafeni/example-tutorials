# Module 01 — MLP on MNIST (PyTorch)

The simplest possible deep learning model: a fully-connected network (no
convolutions) classifying handwritten digits. Use this as the first
module in the course — it establishes the training-loop pattern every
later module reuses.

- **Kernel type:** PyTorch (GPU)
- **Dataset:** MNIST, downloads automatically (~10 MB) on first run
- **Extra packages:** none beyond what the kernel already has
- **Est. GPU time:** ~3 minutes for 5 epochs

## Step-by-step: JupyterLab

1. Spawn or switch your JupyterLab session to a **PyTorch (GPU)** kernel
   (see the [top-level README](../README.md) if you haven't cloned this
   repo yet).
2. In a JupyterLab terminal:
   ```bash
   cd ~/ml-platform/docs/demos/student-examples/01-mlp-mnist-pytorch
   ```
3. Open `notebook.ipynb` from the file browser, confirm the kernel shown
   top-right is a **PyTorch** kernel, then `Run` → `Run All Cells`.

## Step-by-step: VS Code or Terminal

This folder also has `script.py` — the identical walkthrough as
`notebook.ipynb`, runnable via the platform's browser VS Code
("VS Code ↗") or browser Terminal ("Terminal ↗"):

```bash
cd ~/ml-platform/docs/demos/student-examples/01-mlp-mnist-pytorch
python script.py
```

**Read the storage warning in the [top-level README's VS Code/Terminal
section](../README.md#step-by-step-vs-code-or-terminal) first** — work
here does not survive a kernel stop/restart unless you click
"Sync to Home."

## What you'll learn

- Writing a PyTorch training loop from scratch (forward pass, loss,
  backward pass, optimizer step)
- Moving data and models onto a GPU with `.to(device)`
- Reading a loss/accuracy curve to judge whether a model is learning

The material ends with four numbered exercises (network size, dropout,
tracking train accuracy, and an optimizer swap) plus an empty cell/section
for you to try them in.
