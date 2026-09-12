"""
Module 09 - NLP Text Classification (TensorFlow) - VS Code / plain-script version.

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
# # Module 09 — Text Classification with an Embedding + LSTM (TensorFlow)
#
# **Goal:** the TensorFlow counterpart to Module 05 — classify news
# headlines into 4 topics (World, Sports, Business, Sci/Tech) with a
# word-embedding + LSTM classifier, using Keras's `TextVectorization` layer
# to handle tokenization for you.
#
# **Before you start:** launch with a **TensorFlow (GPU)** kernel (see this
# module's [README](README.md)).

# %%
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
assert len(gpus) > 0, (
    "No GPU detected in this kernel.\n"
    "Go back to the portal's Servers tab, stop this session, spawn a new one "
    "using the kernel named in this module's README, then run this again."
)
print("GPU(s):", gpus)

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

train_texts, train_labels = raw["train"]["text"], raw["train"]["label"]
test_texts, test_labels = raw["test"]["text"], raw["test"]["label"]

# %% [markdown]
# ## 2. Build a text vectorization layer

# %%
max_len = 40
vocab_size = 20000

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=max_len,
)
vectorize_layer.adapt(train_texts)

print(f"Vocabulary size: {len(vectorize_layer.get_vocabulary())}")

# %% [markdown]
# ## 3. Build tf.data datasets

# %%
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    .shuffle(10000)
    .batch(128)
    .map(lambda x, y: (vectorize_layer(x), y))
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = (
    tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
    .batch(256)
    .map(lambda x, y: (vectorize_layer(x), y))
    .prefetch(tf.data.AUTOTUNE)
)

# %% [markdown]
# ## 4. Define the model

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_len,)),
    tf.keras.layers.Embedding(vocab_size, 128, mask_zero=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(4),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()

# %% [markdown]
# ## 5. Train

# %%
history = model.fit(train_ds, epochs=4, validation_data=test_ds)

# %% [markdown]
# ## 6. Try it on your own headline

# %%
def predict(text):
    logits = model.predict(vectorize_layer(tf.constant([text])), verbose=0)
    return label_names[logits.argmax(axis=1)[0]]

for headline in [
    "Local team wins championship after dramatic overtime finish",
    "Central bank raises interest rates amid inflation concerns",
    "New chip promises twice the performance at half the power",
]:
    print(f"{headline!r} -> {predict(headline)}")

# %% [markdown]
# ## Exercises
#
# 1. **Bidirectional LSTM.** Wrap the LSTM in
#    `tf.keras.layers.Bidirectional(...)` — does accuracy improve?
# 2. **Try GRU instead of LSTM** (`tf.keras.layers.GRU`, same interface) —
#    does it train faster? Does accuracy change?
# 3. **Vocabulary size.** Try `vocab_size = 5000` and `vocab_size = 50000`
#    — how does that affect accuracy and embedding-layer memory use?
# 4. **Stretch goal:** plot a confusion matrix
#    (`sklearn.metrics.confusion_matrix`) over the 4 classes, and compare
#    directly with Module 05's PyTorch version — same task, same dataset,
#    different framework.

# %%
# Your experiments here
