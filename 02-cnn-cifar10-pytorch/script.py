"""
Module 02 - CNN on CIFAR-10 (PyTorch) - VS Code / plain-script version.

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
# # Module 02 — CNN Image Classifier on CIFAR-10 (PyTorch)
#
# **Goal:** train a convolutional neural network on real (small) color
# photos — CIFAR-10 has 10 classes (airplane, car, bird, cat, ...), 32x32
# color images.
#
# **Before you start:** launch with a **PyTorch (GPU)** kernel — see this
# module's [README](README.md).

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
# ## 1. Load CIFAR-10 with data augmentation

# %%
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
classes = train_ds.classes

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

print(f"Train: {len(train_ds)}  Test: {len(test_ds)}  Classes: {classes}")

# %% [markdown]
# ## 2. Define a small CNN

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 32 -> 16
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 16 -> 8
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)   # 8 -> 4
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

model = SmallCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(sum(p.numel() for p in model.parameters()), "parameters")

# %% [markdown]
# ## 3. Train

# %%
num_epochs = 10
history = {"train_loss": [], "test_acc": []}

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    test_acc = correct / len(test_ds)

    history["train_loss"].append(avg_loss)
    history["test_acc"].append(test_acc)
    print(f"epoch {epoch}/{num_epochs}  train_loss={avg_loss:.4f}  test_acc={test_acc:.4f}")

# %% [markdown]
# ## 4. Plot training curves

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history["train_loss"], marker="o")
axes[0].set_title("Training loss")
axes[0].set_xlabel("epoch")

axes[1].plot(history["test_acc"], marker="o", color="green")
axes[1].set_title("Test accuracy")
axes[1].set_xlabel("epoch")
plt.tight_layout()
import os
os.makedirs("results", exist_ok=True)
plt.savefig("results/plot_01.png", dpi=120, bbox_inches="tight")
print("Saved plot to results/plot_01.png")
plt.show()

# %% [markdown]
# ## 5. Per-class accuracy + sample predictions

# %%
import numpy as np

model.eval()
class_correct = np.zeros(10)
class_total = np.zeros(10)
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        for label, pred in zip(labels, preds):
            class_total[label] += 1
            class_correct[label] += (label == pred).item()

for i, cls in enumerate(classes):
    print(f"{cls:>10s}: {class_correct[i] / class_total[i]:.2%}")

images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
    preds = model(images).argmax(dim=1)

def unnormalize(img):
    img = img.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * std + mean, 0, 1)

fig, axes = plt.subplots(1, 8, figsize=(16, 2.5))
for i, ax in enumerate(axes):
    ax.imshow(unnormalize(images[i]))
    pred, true = classes[preds[i]], classes[labels[i]]
    ax.set_title(f"{pred}\n({true})", fontsize=9, color="green" if pred == true else "red")
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
# 1. **Add batch normalization** (`nn.BatchNorm2d`) after each conv layer —
#    does it train faster or reach higher accuracy?
# 2. **Try a larger GPU allocation** (if your platform offers one) and
#    double the batch size (256) — does an epoch get noticeably faster?
# 3. **Confusion matrix.** Use `sklearn.metrics.confusion_matrix` (run
#    `%pip install -q scikit-learn` first if needed) to see which classes get
#    confused with each other most.
# 4. **Stretch goal:** compare this to Module 03's TensorFlow version of the
#    same task — same architecture, same dataset. Which framework did you
#    find easier to write? Which trained faster on your GPU slice?

# %%
# Your experiments here
