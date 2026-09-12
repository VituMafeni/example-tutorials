"""
Module 04 - Transfer Learning: ResNet18 (PyTorch) - VS Code / plain-script version.

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
# # Module 04 — Transfer Learning: Fine-Tune ResNet18 (PyTorch)
#
# **Goal:** instead of training a CNN from scratch (Module 02), start from a
# ResNet18 already trained on ImageNet and adapt it to CIFAR-10. This is how
# most real-world image classifiers actually get built.
#
# **Before you start:** launch with a **PyTorch (GPU)** kernel. Resizing
# images up for ResNet uses more memory than Modules 02/03 — if you hit a
# CUDA out-of-memory error, try a smaller batch size first, or ask your
# admin about a kernel with more GPU memory.

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
# ## 0. Make sure torchvision is available

# %%
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchvision"])

# %% [markdown]
# ## 1. Load CIFAR-10, resized for ResNet

# %%
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

imagenet_mean, imagenet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
classes = train_ds.classes

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

print(f"Train: {len(train_ds)}  Test: {len(test_ds)}")

# %% [markdown]
# ## 2. Load a pretrained ResNet18 and freeze it

# %%
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Total parameters:     {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## 3. Stage 1 — train only the new final layer

# %%
import torch.nn.functional as F

def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(test_ds)

history = {"stage1_acc": [], "stage2_acc": []}

for epoch in range(1, 4):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    acc = evaluate()
    history["stage1_acc"].append(acc)
    print(f"[stage 1] epoch {epoch}/3  loss={running_loss / len(train_loader):.4f}  test_acc={acc:.4f}")

# %% [markdown]
# ## 4. Stage 2 — unfreeze the last block and fine-tune

# %%
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4
)

for epoch in range(1, 3):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    acc = evaluate()
    history["stage2_acc"].append(acc)
    print(f"[stage 2] epoch {epoch}/2  loss={running_loss / len(train_loader):.4f}  test_acc={acc:.4f}")

# %% [markdown]
# ## 5. Compare stage 1 vs stage 2

# %%
import matplotlib.pyplot as plt

all_acc = history["stage1_acc"] + history["stage2_acc"]
stage_boundary = len(history["stage1_acc"])

plt.plot(range(1, len(all_acc) + 1), all_acc, marker="o")
plt.axvline(stage_boundary + 0.5, color="gray", linestyle="--", label="unfroze layer4")
plt.xlabel("epoch")
plt.ylabel("test accuracy")
plt.legend()
plt.title("Transfer learning: frozen backbone -> fine-tuned")
import os
os.makedirs("results", exist_ok=True)
plt.savefig("results/plot_01.png", dpi=120, bbox_inches="tight")
print("Saved plot to results/plot_01.png")
plt.show()

print(f"Best accuracy with only the new layer trained: {max(history['stage1_acc']):.4f}")
print(f"Best accuracy after fine-tuning layer4:         {max(history['stage2_acc']):.4f}")

# %% [markdown]
# ## Exercises
#
# 1. **Compare to training from scratch.** In Module 02 your CNN reached some
#    test accuracy after 10 epochs of full training. How does that compare to
#    just 3 epochs of training a single new layer here?
# 2. **Unfreeze more.** Also unfreeze `model.layer3` in Stage 2 — does
#    accuracy improve further, or does it start overfitting?
# 3. **Try a different backbone.** Swap `resnet18` for `resnet34` (same API)
#    — is the accuracy gain worth the extra training time on your GPU slice?
# 4. **Stretch goal:** skip Stage 1 training entirely and just run
#    `evaluate()` right after freezing — this shows how much of the accuracy
#    comes from the pretrained features alone, before you fine-tune anything.

# %%
# Your experiments here
