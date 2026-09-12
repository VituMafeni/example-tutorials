# Module 04 — Transfer Learning: Fine-Tune ResNet18 (PyTorch)

Instead of training a CNN from scratch (Module 02), start from a ResNet18
already trained on ImageNet and adapt it to CIFAR-10 in two stages: first
train only a new final layer, then unfreeze and fine-tune the last block.
This is how most real-world image classifiers actually get built.

- **Kernel type:** PyTorch (GPU) — resizing images up for ResNet uses more
  memory than Modules 02/03; if you hit a CUDA out-of-memory error, try a
  smaller batch size first, or ask your admin about a kernel with more
  GPU memory
- **Dataset:** CIFAR-10, downloads automatically (~170 MB) on first run
- **Extra packages:** `torchvision` — installed by the first cell/step
  (`pip install -q torchvision`)
- **Est. GPU time:** ~10 minutes for all 5 epochs (3 + 2)

## Step-by-step: JupyterLab

1. Spawn or switch your JupyterLab session to a **PyTorch (GPU)** kernel
   (see the [top-level README](../README.md) if you haven't cloned this
   repo yet).
2. In a JupyterLab terminal:
   ```bash
   cd ~/ml-platform/docs/demos/student-examples/04-transfer-learning-resnet-pytorch
   ```
3. Open `notebook.ipynb` from the file browser, confirm the kernel shown
   top-right is a **PyTorch** kernel, then `Run` → `Run All Cells`.

## Step-by-step: VS Code or Terminal

This folder also has `script.py` — the identical walkthrough as
`notebook.ipynb`, runnable via the platform's browser VS Code
("VS Code ↗") or browser Terminal ("Terminal ↗"):

```bash
cd ~/ml-platform/docs/demos/student-examples/04-transfer-learning-resnet-pytorch
python script.py
```

Plots save as PNG files under `results/` instead of displaying inline.
**Read the storage warning in the [top-level README's VS Code/Terminal
section](../README.md#step-by-step-vs-code-or-terminal) first** — work
here does not survive a kernel stop/restart unless you click
"Sync to Home."

## What you'll learn

- Loading a pretrained model (`torchvision.models.resnet18`) and swapping
  its final classification layer
- Freezing parameters with `requires_grad = False` and selectively
  unfreezing them later
- Two-stage fine-tuning: cheap head-only training, then a low-learning-rate
  pass over part of the backbone

The material ends with four numbered exercises (comparing to Module 02's
from-scratch result, unfreezing more layers, trying `resnet34`, and
measuring the pretrained-features-only baseline) plus an empty cell/section
for you to try them in.

**Do Module 02 first** if you can — Exercise 1 asks you to compare results
directly against it.
