"""
Module 05 - NLP Text Classification (PyTorch) - VS Code / plain-script version.

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
# # Module 05 — Text Classification with an Embedding + LSTM (PyTorch)
#
# **Goal:** classify news headlines into 4 topics (World, Sports, Business,
# Sci/Tech) using a word-embedding + LSTM classifier — your first NLP deep
# learning model.
#
# **Before you start:** launch with a **PyTorch (GPU)** kernel (see this
# module's [README](README.md)).

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
# ## 0. Install the dataset library

# %%
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])

# %% [markdown]
# ## 1. Load the AG News dataset

# %%
from datasets import load_dataset

raw = load_dataset("ag_news")
label_names = raw["train"].features["label"].names
print(label_names)
print(raw["train"][0])

# %% [markdown]
# ## 2. Build a vocabulary

# %%
import re
from collections import Counter

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

counter = Counter()
for example in raw["train"]:
    counter.update(tokenize(example["text"]))

vocab_size = 20000
most_common = counter.most_common(vocab_size - 2)
word2idx = {"<pad>": 0, "<unk>": 1}
for word, _ in most_common:
    word2idx[word] = len(word2idx)

print(f"Vocabulary size: {len(word2idx)}")

def encode(text, max_len=40):
    ids = [word2idx.get(tok, 1) for tok in tokenize(text)][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

# %% [markdown]
# ## 3. Build PyTorch datasets

# %%
import torch
from torch.utils.data import Dataset, DataLoader

class NewsDataset(Dataset):
    def __init__(self, split):
        self.texts = split["text"]
        self.labels = split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = torch.tensor(encode(self.texts[idx]), dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return ids, label

train_ds = NewsDataset(raw["train"])
test_ds = NewsDataset(raw["test"])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
print(f"Train: {len(train_ds)}  Test: {len(test_ds)}")

# %% [markdown]
# ## 4. Define the model

# %%
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden.squeeze(0))

model = LSTMClassifier(len(word2idx)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(model)

# %% [markdown]
# ## 5. Train

# %%
import torch.nn.functional as F

num_epochs = 4
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for ids, labels in train_loader:
        ids, labels = ids.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(ids), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = 0
    with torch.no_grad():
        for ids, labels in test_loader:
            ids, labels = ids.to(device), labels.to(device)
            correct += (model(ids).argmax(dim=1) == labels).sum().item()
    test_acc = correct / len(test_ds)
    print(f"epoch {epoch}/{num_epochs}  train_loss={running_loss / len(train_loader):.4f}  test_acc={test_acc:.4f}")

# %% [markdown]
# ## 6. Try it on your own headline

# %%
def predict(text):
    model.eval()
    ids = torch.tensor([encode(text)], dtype=torch.long).to(device)
    with torch.no_grad():
        pred = model(ids).argmax(dim=1).item()
    return label_names[pred]

for headline in [
    "Local team wins championship after dramatic overtime finish",
    "Central bank raises interest rates amid inflation concerns",
    "New chip promises twice the performance at half the power",
]:
    print(f"{headline!r} -> {predict(headline)}")

# %% [markdown]
# ## Exercises
#
# 1. **Bidirectional LSTM.** Set `bidirectional=True` in `nn.LSTM` (remember
#    to double the input size of `self.fc` and concatenate both directions'
#    final hidden states).
# 2. **Try GRU instead of LSTM** (`nn.GRU`, same interface) — does it train
#    faster? Does accuracy change?
# 3. **Vocabulary size.** Try `vocab_size = 5000` and `vocab_size = 50000` —
#    how does that affect accuracy and embedding-layer memory use?
# 4. **Stretch goal:** print a confusion matrix
#    (`sklearn.metrics.confusion_matrix`) over the 4 classes — which two
#    topics does the model confuse most, and why do you think that is (look
#    at a few misclassified headlines)?

# %%
# Your experiments here
