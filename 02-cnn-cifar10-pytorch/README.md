# Module 02 — CNN Image Classifier on CIFAR-10 (PyTorch)

A small convolutional network on real (if tiny) color photos — 10 classes
like airplane, cat, and truck. Pairs with Module 03, which solves the exact
same task in TensorFlow.

- **Kernel type:** PyTorch (GPU)
- **Dataset:** CIFAR-10, downloads automatically (~170 MB) on first run
- **Extra packages:** `torchvision` — installed by the first cell/step
  (`pip install -q torchvision`)
- **Est. GPU time:** ~8 minutes for 10 epochs

## Step-by-step: JupyterLab

1. Spawn or switch your JupyterLab session to a **PyTorch (GPU)** kernel
   (see the [top-level README](../README.md) if you haven't cloned this
   repo yet).
2. In a JupyterLab terminal:
   ```bash
   cd ~/ml-platform/docs/demos/student-examples/02-cnn-cifar10-pytorch
   ```
3. Open `notebook.ipynb` from the file browser, confirm the kernel shown
   top-right is a **PyTorch** kernel, then `Run` → `Run All Cells`.

## Step-by-step: VS Code or Terminal

This folder also has `script.py` — the identical walkthrough as
`notebook.ipynb`, runnable via the platform's browser VS Code
("VS Code ↗") or browser Terminal ("Terminal ↗"):

```bash
cd ~/ml-platform/docs/demos/student-examples/02-cnn-cifar10-pytorch
python script.py
```

Plots save as PNG files under `results/` instead of displaying inline.
**Read the storage warning in the [top-level README's VS Code/Terminal
section](../README.md#step-by-step-vs-code-or-terminal) first** — work
here does not survive a kernel stop/restart unless you click
"Sync to Home."

## What you'll learn

- Building a multi-layer CNN (`Conv2d` + `MaxPool2d` stacked three times)
- Data augmentation (`RandomCrop`, `RandomHorizontalFlip`) and why it helps
  on a small dataset
- Per-class accuracy, not just overall accuracy
- Reading normalized image tensors back into a viewable image

The material ends with four numbered exercises (batch norm, a larger GPU
allocation, a confusion matrix, and a framework comparison with Module 03)
plus an empty cell/section for you to try them in.
