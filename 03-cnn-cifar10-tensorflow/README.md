# Module 03 — CNN Image Classifier on CIFAR-10 (TensorFlow)

The exact same task as Module 02 — same dataset, same-size CNN — written
in TensorFlow/Keras instead of PyTorch, so you can compare the two
frameworks directly on identical ground.

- **Kernel type:** TensorFlow (GPU)
- **Dataset:** CIFAR-10, downloads automatically (~170 MB) on first run
- **Extra packages:** none beyond what the kernel already has
- **Est. GPU time:** ~8 minutes for 10 epochs

## Step-by-step: JupyterLab

1. Spawn or switch your JupyterLab session to a **TensorFlow (GPU)**
   kernel (see the [top-level README](../README.md) if you haven't
   cloned this repo yet).
2. In a JupyterLab terminal:
   ```bash
   cd ~/ml-platform/docs/demos/student-examples/03-cnn-cifar10-tensorflow
   ```
3. Open `notebook.ipynb` from the file browser, confirm the kernel shown
   top-right is a **TensorFlow** kernel, then `Run` → `Run All Cells`.

## Step-by-step: VS Code or Terminal

This folder also has `script.py` — the identical walkthrough as
`notebook.ipynb`, runnable via the platform's browser VS Code
("VS Code ↗") or browser Terminal ("Terminal ↗"):

```bash
cd ~/ml-platform/docs/demos/student-examples/03-cnn-cifar10-tensorflow
python script.py
```

Plots save as PNG files under `results/` instead of displaying inline.
**Read the storage warning in the [top-level README's VS Code/Terminal
section](../README.md#step-by-step-vs-code-or-terminal) first** — work
here does not survive a kernel stop/restart unless you click
"Sync to Home."

## What you'll learn

- Building the same CNN architecture as Module 02 with `tf.keras.Sequential`
- `model.compile()` / `model.fit()` versus a hand-written training loop
- Reading the `History` object Keras returns for plotting curves

The material ends with four numbered exercises (batch norm, early
stopping, data augmentation layers, and a framework comparison with
Module 02) plus an empty cell/section for you to try them in.

**Do Module 02 first** if you haven't — the point of this module is the
side-by-side comparison, and it'll mean more once you've seen the PyTorch
version's training loop.
