"""
Module 08 - Transfer Learning: MobileNetV2 (TensorFlow) - VS Code / plain-script version.

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
# # Module 08 — Transfer Learning: Fine-Tune MobileNetV2 (TensorFlow)
#
# **Goal:** the TensorFlow counterpart to Module 04 — start from a
# MobileNetV2 already trained on ImageNet and adapt it to CIFAR-10 in two
# stages: first train only a new classification head, then unfreeze and
# fine-tune the top of the backbone.
#
# **Before you start:** launch with a **TensorFlow (GPU)** kernel. Resizing
# images up for MobileNetV2 uses more memory than Module 03 — if you hit an
# out-of-memory error, try a smaller batch size first, or ask your admin
# about a kernel with more GPU memory.

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
# ## 1. Load CIFAR-10, resized for MobileNetV2

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()
classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

IMG_SIZE = 128

def preprocess(images, labels):
    images = tf.image.resize(images, (IMG_SIZE, IMG_SIZE))
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000)
    .batch(64)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .batch(128)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

print(f"Train: {len(x_train)}  Test: {len(x_test)}")

# %% [markdown]
# ## 2. Load a pretrained MobileNetV2 and freeze it

# %%
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters:     {model.count_params():,}")

# %% [markdown]
# ## 3. Stage 1 — train only the new classification head

# %%
history_stage1 = model.fit(train_ds, epochs=3, validation_data=test_ds)

# %% [markdown]
# ## 4. Stage 2 — unfreeze the top of the backbone and fine-tune

# %%
base_model.trainable = True
# Keep the bottom of the network frozen; only fine-tune from this layer up.
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history_stage2 = model.fit(train_ds, epochs=2, validation_data=test_ds)

# %% [markdown]
# ## 5. Compare stage 1 vs stage 2

# %%
import matplotlib.pyplot as plt

all_acc = history_stage1.history["val_accuracy"] + history_stage2.history["val_accuracy"]
stage_boundary = len(history_stage1.history["val_accuracy"])

plt.plot(range(1, len(all_acc) + 1), all_acc, marker="o")
plt.axvline(stage_boundary + 0.5, color="gray", linestyle="--", label="unfroze top of backbone")
plt.xlabel("epoch")
plt.ylabel("test accuracy")
plt.legend()
plt.title("Transfer learning: frozen backbone -> fine-tuned")
import os
os.makedirs("results", exist_ok=True)
plt.savefig("results/plot_01.png", dpi=120, bbox_inches="tight")
print("Saved plot to results/plot_01.png")
plt.show()

print(f"Best accuracy with only the head trained:     {max(history_stage1.history['val_accuracy']):.4f}")
print(f"Best accuracy after fine-tuning the backbone: {max(history_stage2.history['val_accuracy']):.4f}")

# %% [markdown]
# ## Exercises
#
# 1. **Compare to training from scratch.** In Module 03 your CNN reached
#    some test accuracy after 10 epochs of full training. How does that
#    compare to just 3 epochs of training a new head here?
# 2. **Unfreeze more.** Change `fine_tune_at` to unfreeze more (or fewer)
#    layers of the backbone — does accuracy improve further, or does it
#    start overfitting?
# 3. **Try a different backbone.** Swap `MobileNetV2` for
#    `tf.keras.applications.ResNet50` (same API, different
#    `preprocess_input`) — is the accuracy gain worth the extra training
#    time?
# 4. **Stretch goal:** skip Stage 1 training entirely and call
#    `model.evaluate(test_ds)` right after freezing — this shows how much
#    of the accuracy comes from the pretrained features alone.

# %%
# Your experiments here
