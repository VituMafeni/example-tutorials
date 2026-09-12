"""
Module 03 - CNN on CIFAR-10 (TensorFlow) - VS Code / plain-script version.

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
# # Module 03 — CNN Image Classifier on CIFAR-10 (TensorFlow)
#
# **Goal:** the exact same task as Module 02 (same dataset, same architecture
# size) but written in TensorFlow/Keras — a chance to compare the two most
# common deep learning frameworks side by side.
#
# **Before you start:** launch with a **TensorFlow (GPU)** kernel — see
# this module's [README](README.md).

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
# ## 1. Load CIFAR-10

# %%
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0

classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

print(f"Train: {x_train.shape}  Test: {x_test.shape}")

# %% [markdown]
# ## 2. Define the same CNN architecture as Module 02

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
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
    epochs=10,
    validation_data=(x_test, y_test),
)

# %% [markdown]
# ## 4. Plot training curves

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
# ## 5. Sample predictions

# %%
probs = tf.nn.softmax(model.predict(x_test[:8])).numpy()
preds = probs.argmax(axis=1)

fig, axes = plt.subplots(1, 8, figsize=(16, 2.5))
for i, ax in enumerate(axes):
    ax.imshow(x_test[i])
    pred, true = classes[preds[i]], classes[y_test[i]]
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
# 1. **Add batch normalization** (`tf.keras.layers.BatchNormalization()`)
#    after each conv layer.
# 2. **Try `tf.keras.callbacks.EarlyStopping`** to stop training automatically
#    once validation loss stops improving.
# 3. **Data augmentation:** add a `tf.keras.layers.RandomFlip("horizontal")` +
#    `RandomCrop` preprocessing layer before the first conv layer (see
#    Module 02's PyTorch augmentation for reference) and compare test
#    accuracy.
# 4. **Stretch goal:** compare `model.fit()`'s epoch time here with Module
#    02's PyTorch loop time on the same GPU tier. Which framework was less
#    code? Which was faster on your GPU slice?

# %%
# Your experiments here
