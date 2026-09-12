# Module 09 — Text Classification with an Embedding + LSTM (TensorFlow)

The TensorFlow counterpart to Module 05 — classify news headlines into 4
topics (World, Sports, Business, Sci/Tech) with a word-embedding + LSTM
classifier, using Keras's `TextVectorization` layer to handle tokenization
for you (rather than the hand-rolled vocabulary in Module 05).

- **Kernel type:** TensorFlow (GPU)
- **Dataset:** AG News, downloaded via the HuggingFace `datasets` library
  on first run (~30 MB)
- **Extra packages:** `datasets` — installed by the first cell/step
  (`pip install -q datasets`)
- **Est. GPU time:** ~6 minutes for 4 epochs

## Step-by-step: JupyterLab

1. Spawn or switch your JupyterLab session to a **TensorFlow (GPU)**
   kernel (see the [top-level README](../README.md) if you haven't
   cloned this repo yet).
2. In a JupyterLab terminal:
   ```bash
   cd ~/ml-platform/docs/demos/student-examples/09-nlp-text-classification-tensorflow
   ```
3. Open `notebook.ipynb` from the file browser, confirm the kernel shown
   top-right is a **TensorFlow** kernel, then `Run` → `Run All Cells`.

## Step-by-step: VS Code or Terminal

This folder also has `script.py` — the identical walkthrough as
`notebook.ipynb`, runnable via the platform's browser VS Code
("VS Code ↗") or browser Terminal ("Terminal ↗"):

```bash
cd ~/ml-platform/docs/demos/student-examples/09-nlp-text-classification-tensorflow
python script.py
```

**Read the storage warning in the [top-level README's VS Code/Terminal
section](../README.md#step-by-step-vs-code-or-terminal) first** — work
here does not survive a kernel stop/restart unless you click
"Sync to Home."

## What you'll learn

- `tf.keras.layers.TextVectorization` — Keras's built-in tokenizer/vocab
  layer, versus building one by hand (see Module 05)
- Building a text input pipeline with `tf.data.Dataset`
- `Embedding` + `LSTM` for sequence classification in Keras
- Running your trained model interactively on a sentence you type yourself

The material ends with four numbered exercises (bidirectional LSTM, GRU
swap, vocabulary size, and a confusion matrix + framework comparison with
Module 05) plus an empty cell/section for you to try them in.

**Do Module 05 first** if you can — the exercises ask you to compare
results directly against it.
