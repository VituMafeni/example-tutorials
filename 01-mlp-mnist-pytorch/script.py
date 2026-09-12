"""
Module 01 - MLP on MNIST (PyTorch) - VS Code / plain-script version.

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
# # Module 01 — MLP on MNIST (PyTorch)
#
# **Goal:** train your first neural network — a fully-connected network (no
# convolutions yet) that classifies handwritten digits.
#
# **Before you start:** make sure you launched this session with a
# **PyTorch (GPU)** kernel (see this module's [README](README.md) if
# you have not). Run the cells top to bottom.

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
# ## 1. Load the MNIST dataset

# %%
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

print(f"Train examples: {len(train_ds)}, Test examples: {len(test_ds)}")

# %% [markdown]
# ## 2. Define the model — a simple MLP

# %%
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(model)

# %% [markdown]
# ## 3. Train

# %%
num_epochs = 5
history = {"train_loss": [], "test_acc": []}

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
    test_acc = correct / len(test_ds)

    history["train_loss"].append(avg_loss)
    history["test_acc"].append(test_acc)
    print(f"epoch {epoch}/{num_epochs}  train_loss={avg_loss:.4f}  test_acc={test_acc:.4f}")

# %% [markdown]
# ## 4. Plot the training curve

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
# ## 5. Look at some predictions

# %%
model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
    preds = model(images).argmax(dim=1)

fig, axes = plt.subplots(1, 8, figsize=(14, 2))
for i, ax in enumerate(axes):
    ax.imshow(images[i].cpu().squeeze(), cmap="gray")
    pred, true = preds[i].item(), labels[i].item()
    ax.set_title(f"pred {pred}\ntrue {true}", fontsize=9, color="green" if pred == true else "red")
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
# 1. **Change the network size.** Try `hidden_size=64` and `hidden_size=512` —
#    how does test accuracy and training time change?
# 2. **Add dropout.** Add `nn.Dropout(0.3)` between the hidden layers. Does it
#    help or hurt with only 5 epochs?
# 3. **Track more.** Modify the training loop to also record *train* accuracy
#    each epoch (not just loss), and add it to the plot.
# 4. **Stretch goal:** swap `Adam` for
#    `torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`. Does it
#    converge as fast in 5 epochs?
#
# Use the empty cell below for your experiments.

# %%
# Your experiments here
