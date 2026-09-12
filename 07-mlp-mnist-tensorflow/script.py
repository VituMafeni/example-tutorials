"""
Module 07 - MLP on MNIST (TensorFlow) - VS Code / plain-script version.

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
# # Module 07 — MLP on MNIST (TensorFlow)
#
# **Goal:** the TensorFlow counterpart to Module 01 — the same first neural
# network (a fully-connected network, no convolutions), classifying
# handwritten digits, so you can compare the two frameworks on the simplest
# possible model.
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
# ## 1. Load the MNIST dataset

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0

print(f"Train: {x_train.shape}  Test: {x_test.shape}")

# %% [markdown]
# ## 2. Define the model — a simple MLP

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(10),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()

# %% [markdown]
# ## 3. Train

# %%
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_data=(x_test, y_test),
)

# %% [markdown]
# ## 4. Plot the training curve

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history.history["loss"], marker="o", label="train")
axes[0].plot(history.history["val_loss"], marker="o", label="test")
axes[0].set_title("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], marker="o", label="train")
axes[1].plot(history.history["val_accuracy"], marker="o", label="test")
axes[1].set_title("Accuracy")
axes[1].legend()
plt.tight_layout()
import os
os.makedirs("results", exist_ok=True)
plt.savefig("results/plot_01.png", dpi=120, bbox_inches="tight")
print("Saved plot to results/plot_01.png")
plt.show()

# %% [markdown]
# ## 5. Look at some predictions

# %%
probs = tf.nn.softmax(model.predict(x_test[:8])).numpy()
preds = probs.argmax(axis=1)

fig, axes = plt.subplots(1, 8, figsize=(14, 2))
for i, ax in enumerate(axes):
    ax.imshow(x_test[i], cmap="gray")
    pred, true = preds[i], y_test[i]
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
# 1. **Change the network size.** Try 64 units and 512 units per layer —
#    how does test accuracy and training time change?
# 2. **Add dropout.** Add `tf.keras.layers.Dropout(0.3)` between the hidden
#    layers. Does it help or hurt with only 5 epochs?
# 3. **Track more.** Inspect `history.history` — what other metrics could
#    you log with a custom `tf.keras.callbacks.Callback`?
# 4. **Stretch goal:** compare this notebook's code and training time
#    directly against Module 01's PyTorch version — which was less code?
#    Which trained faster on your GPU?

# %%
# Your experiments here
