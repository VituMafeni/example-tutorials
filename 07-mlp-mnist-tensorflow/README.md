# Module 07 — MLP on MNIST (TensorFlow)

The TensorFlow counterpart to Module 01 — the same first neural network
(a fully-connected network, no convolutions), classifying handwritten
digits, written in `tf.keras` instead of PyTorch.

- **Kernel type:** TensorFlow (GPU)
- **Dataset:** MNIST, downloads automatically (~10 MB) on first run
- **Extra packages:** none beyond what the kernel already has
- **Est. GPU time:** ~3 minutes for 5 epochs

## Step-by-step: JupyterLab

1. Spawn or switch your JupyterLab session to a **TensorFlow (GPU)**
   kernel (see the [top-level README](../README.md) if you haven't
   cloned this repo yet).
2. In a JupyterLab terminal:
   ```bash
   cd ~/ml-platform/docs/demos/student-examples/07-mlp-mnist-tensorflow
   ```
3. Open `notebook.ipynb` from the file browser, confirm the kernel shown
   top-right is a **TensorFlow** kernel, then `Run` → `Run All Cells`.

## Step-by-step: VS Code or Terminal

This folder also has `script.py` — the identical walkthrough as
`notebook.ipynb`, runnable via the platform's browser VS Code
("VS Code ↗") or browser Terminal ("Terminal ↗"):

```bash
cd ~/ml-platform/docs/demos/student-examples/07-mlp-mnist-tensorflow
python script.py
```

**Read the storage warning in the [top-level README's VS Code/Terminal
section](../README.md#step-by-step-vs-code-or-terminal) first** — work
here does not survive a kernel stop/restart unless you click
"Sync to Home."

## What you'll learn

- Building and training a model with `tf.keras.Sequential` and
  `model.fit()`
- Reading the `History` object Keras returns for plotting curves
- How the same MLP looks noticeably different in TensorFlow vs. Module
  01's hand-written PyTorch training loop

The material ends with four numbered exercises (network size, dropout,
tracking more metrics, and a framework comparison with Module 01) plus an
empty cell/section for you to try them in.

**Do Module 01 first** if you haven't — the point of this module is the
side-by-side comparison.
