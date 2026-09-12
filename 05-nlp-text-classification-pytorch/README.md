# Module 05 — Text Classification with an Embedding + LSTM (PyTorch)

Your first NLP deep learning model: classify news headlines into 4 topics
(World, Sports, Business, Sci/Tech) with a word-embedding + LSTM
classifier you build and train yourself (no pretrained language model).

- **Kernel type:** PyTorch (GPU)
- **Dataset:** AG News, downloaded via the HuggingFace `datasets` library
  on first run (~30 MB)
- **Extra packages:** `datasets` — installed by the first cell/step
  (`pip install -q datasets`)
- **Est. GPU time:** ~6 minutes for 4 epochs

## Step-by-step: JupyterLab

1. Spawn or switch your JupyterLab session to a **PyTorch (GPU)** kernel
   (see the [top-level README](../README.md) if you haven't cloned this
   repo yet).
2. In a JupyterLab terminal:
   ```bash
   cd ~/ml-platform/docs/demos/student-examples/05-nlp-text-classification-pytorch
   ```
3. Open `notebook.ipynb` from the file browser, confirm the kernel shown
   top-right is a **PyTorch** kernel, then `Run` → `Run All Cells`.

## Step-by-step: VS Code or Terminal

This folder also has `script.py` — the identical walkthrough as
`notebook.ipynb`, runnable via the platform's browser VS Code
("VS Code ↗") or browser Terminal ("Terminal ↗"):

```bash
cd ~/ml-platform/docs/demos/student-examples/05-nlp-text-classification-pytorch
python script.py
```

**Read the storage warning in the [top-level README's VS Code/Terminal
section](../README.md#step-by-step-vs-code-or-terminal) first** — work
here does not survive a kernel stop/restart unless you click
"Sync to Home."

## What you'll learn

- Turning raw text into fixed-length integer sequences (tokenize, build a
  vocabulary, pad/truncate)
- `nn.Embedding` followed by `nn.LSTM` for sequence classification
- Running your trained model interactively on a sentence you type yourself

The material ends with four numbered exercises (bidirectional LSTM, GRU
swap, vocabulary size, and a confusion matrix) plus an empty cell/section
for you to try them in.
