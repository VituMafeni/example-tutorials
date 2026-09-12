# Module 08 — Transfer Learning: Fine-Tune MobileNetV2 (TensorFlow)

The TensorFlow counterpart to Module 04 — start from a MobileNetV2
already trained on ImageNet and adapt it to CIFAR-10 in two stages: first
train only a new classification head, then unfreeze and fine-tune the top
of the backbone.

- **Kernel type:** TensorFlow (GPU) — resizing images up for MobileNetV2
  uses more memory than Module 03; if you hit an out-of-memory error, try
  a smaller batch size first, or ask your admin about a kernel with more
  GPU memory
- **Dataset:** CIFAR-10, downloads automatically (~170 MB) on first run
- **Extra packages:** none beyond what the kernel already has
- **Est. GPU time:** ~10 minutes for all 5 epochs (3 + 2)

## Step-by-step: JupyterLab

1. Spawn or switch your JupyterLab session to a **TensorFlow (GPU)**
   kernel (see the [top-level README](../README.md) if you haven't
   cloned this repo yet).
2. In a JupyterLab terminal:
   ```bash
   cd ~/ml-platform/docs/demos/student-examples/08-transfer-learning-mobilenet-tensorflow
   ```
3. Open `notebook.ipynb` from the file browser, confirm the kernel shown
   top-right is a **TensorFlow** kernel, then `Run` → `Run All Cells`.

## Step-by-step: VS Code or Terminal

This folder also has `script.py` — the identical walkthrough as
`notebook.ipynb`, runnable via the platform's browser VS Code
("VS Code ↗") or browser Terminal ("Terminal ↗"):

```bash
cd ~/ml-platform/docs/demos/student-examples/08-transfer-learning-mobilenet-tensorflow
python script.py
```

Plots save as PNG files under `results/` instead of displaying inline.
**Read the storage warning in the [top-level README's VS Code/Terminal
section](../README.md#step-by-step-vs-code-or-terminal) first** — work
here does not survive a kernel stop/restart unless you click
"Sync to Home."

## What you'll learn

- Loading a pretrained model (`tf.keras.applications.MobileNetV2`) and
  attaching a new classification head
- Freezing/unfreezing layers with `.trainable` and why you must
  `model.compile()` again after changing it
- Building an input pipeline with `tf.data.Dataset` (`.map`, `.batch`,
  `.prefetch`)
- Two-stage fine-tuning: cheap head-only training, then a low-learning-rate
  pass over part of the backbone

The material ends with four numbered exercises (comparing to Module 03's
from-scratch result, unfreezing more layers, trying `ResNet50`, and
measuring the pretrained-features-only baseline) plus an empty cell/section
for you to try them in.

**Do Module 03 first** if you can — Exercise 1 asks you to compare results
directly against it. This also pairs with Module 04, the same idea in
PyTorch with ResNet18.
