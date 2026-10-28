# Module 10 — Training and Saving a Model on MNIST (PyTorch)

A small fully-connected network trained on handwritten digits, with the
trained weights saved to disk at the end. Where [Module 01](../01-mlp-mnist-pytorch/)
focuses on the training loop itself, this module focuses on the output of
training: a `model.pth` file you can load again later without retraining.

- **Kernel type:** PyTorch (GPU recommended; falls back to CPU)
- **Dataset:** MNIST, downloads automatically (~10 MB) to `./data` on first run
- **Extra packages:** none beyond `torch` and `torchvision`
- **Est. GPU time:** ~10–15 minutes for the default 100 epochs

## What the code does

`train.py` runs top to bottom in five sections:

1. **Configuration** — sets `BATCH_SIZE`, `EPOCHS` and `LR` (learning
   rate), and picks the GPU if one is available, otherwise the CPU.
2. **Dataset** — downloads the 60,000 MNIST training images and wraps
   them in a `DataLoader` that serves shuffled batches of 64.
   `transforms.ToTensor()` turns each 28×28 image into a tensor with
   pixel values in `[0, 1]`.
3. **Model** — `SimpleNet` flattens each image into 784 numbers, passes
   them through one hidden layer of 128 neurons with ReLU, and outputs
   10 scores (one per digit).
4. **Training** — for each epoch, every batch goes through the forward
   pass, the cross-entropy loss, the backward pass and an Adam optimizer
   step. The average loss for the epoch is printed:
   ```
   Epoch [1/100] Loss: 0.3xxx
   Epoch [2/100] Loss: 0.1xxx
   ...
   ```
   The loss should drop quickly in the first few epochs and then level off.
5. **Save** — writes the trained weights (`model.state_dict()`) to
   `model.pth` in the current directory.

## How to run

```bash
cd ~/ml-platform/docs/demos/student-examples/10-mnist-checkpoint-pytorch
python train.py
```

If you're running it on your own machine, install the dependencies first:

```bash
pip install torch torchvision
```

When it finishes you'll see `Training complete.` and a new `model.pth`
file next to `train.py`.

**Read the storage warning in the [top-level README's VS Code/Terminal
section](../README.md#step-by-step-vs-code-or-terminal) first** — the
saved `model.pth` does not survive a kernel stop/restart unless you click
"Sync to Home."

### Loading the saved model

The file holds only the weights, so you need the `SimpleNet` class
definition to load it:

```python
import torch
from train import SimpleNet  # or copy the class definition

model = SimpleNet()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()
```

(Importing from `train.py` will rerun the training, because the script has
no `if __name__ == "__main__":` guard — copying the class is quicker.)

## Changing parameters

All the main settings are constants at the top of `train.py`:

```python
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001
```

| Parameter | What it controls | Things to try |
|---|---|---|
| `EPOCHS` | How many full passes over the 60,000 images | `5`–`20` is enough for this model; 100 mostly overfits and takes much longer |
| `BATCH_SIZE` | Images per gradient step | `32` (noisier, more steps), `128` or `256` (faster per epoch on a GPU) |
| `LR` | Size of each optimizer step | `0.01` (faster but may be unstable), `0.0001` (stable but slow) |

Other things you can change directly in the code:

- **Hidden layer size** — change `128` in both `nn.Linear(28 * 28, 128)`
  and `nn.Linear(128, 10)` (e.g. `256` or `512`).
- **Output file name** — change `"model.pth"` in `torch.save(...)`, e.g.
  to include the settings: `f"model_lr{LR}_bs{BATCH_SIZE}.pth"`.
- **Optimizer** — swap `optim.Adam` for
  `optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`.

Change one thing at a time and compare the printed losses, so you know
which change made the difference.

## How to improve training

The script only reports training loss, so the first step is to measure
how well the model actually does on images it hasn't seen.

1. **Add a test set and accuracy.** Load
   `datasets.MNIST(root="./data", train=False, ...)` into a second
   `DataLoader` (`shuffle=False`), and after each epoch run it under
   `model.eval()` and `torch.no_grad()`, counting how often
   `outputs.argmax(1) == labels`. This model typically reaches around
   97–98% test accuracy.
2. **Save the best model, not just the last one.** Keep track of the best
   test accuracy and call `torch.save(...)` only when it improves. With
   100 epochs the last epoch is rarely the best.
3. **Save full checkpoints so training can resume.** Save the optimizer
   state and epoch number alongside the weights:
   ```python
   torch.save({
       "epoch": epoch,
       "model": model.state_dict(),
       "optimizer": optimizer.state_dict(),
   }, "checkpoint.pth")
   ```
   On restart, load it, call `load_state_dict` on both the model and the
   optimizer, and continue the loop from `epoch + 1`.
4. **Stop early.** If test accuracy hasn't improved for, say, 5 epochs,
   `break` out of the loop instead of running all 100.
5. **Normalize the inputs.** Use
   `transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])`
   (MNIST's mean and standard deviation) — this usually speeds up
   learning slightly.
6. **Reduce overfitting.** Add `nn.Dropout(0.2)` after the `nn.ReLU()`,
   or pass `weight_decay=1e-4` to the optimizer.
7. **Use a learning-rate schedule.** For example
   `scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)`
   and call `scheduler.step()` once at the end of each epoch.
8. **Use a bigger or better model.** Add a second hidden layer, or switch
   to a small CNN (see [Module 02](../02-cnn-cifar10-pytorch/)) — CNNs
   reach about 99% on MNIST.
9. **Load data faster on a GPU.** Pass `num_workers=2, pin_memory=True`
   to the `DataLoader`.
