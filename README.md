# Student Examples — GPU Deep Learning Coursework

Nine self-contained, GPU-only deep learning examples for an undergraduate
AI/ML course, built to run directly on this platform — five in PyTorch,
four in TensorFlow. Every module ships as **both**:

- `notebook.ipynb` — a Jupyter notebook, for JupyterLab
- `script.py` — the identical walkthrough as a plain script, for the
  platform's browser-based VS Code or its browser Terminal

Same content, same exercises, either format — pick whichever your course
uses. Every module checks for a GPU in its first few lines and stops with
a clear message if one isn't present — there is no CPU fallback path, by
design.

| # | Module | Framework | Est. GPU time |
|---|---|---|---|
| 01 | [MLP on MNIST](01-mlp-mnist-pytorch/) | PyTorch | ~3 min |
| 02 | [CNN on CIFAR-10](02-cnn-cifar10-pytorch/) | PyTorch | ~8 min |
| 03 | [CNN on CIFAR-10](03-cnn-cifar10-tensorflow/) | TensorFlow | ~8 min |
| 04 | [Transfer learning: ResNet18](04-transfer-learning-resnet-pytorch/) | PyTorch | ~10 min |
| 05 | [NLP text classification](05-nlp-text-classification-pytorch/) | PyTorch | ~6 min |
| 06 | [Deep Q-Network on CartPole](06-rl-dqn-cartpole-pytorch/) | PyTorch | ~10 min |
| 07 | [MLP on MNIST](07-mlp-mnist-tensorflow/) | TensorFlow | ~3 min |
| 08 | [Transfer learning: MobileNetV2](08-transfer-learning-mobilenet-tensorflow/) | TensorFlow | ~10 min |
| 09 | [NLP text classification](09-nlp-text-classification-tensorflow/) | TensorFlow | ~6 min |
| 10 | [Training and saving a model on MNIST](10-mnist-checkpoint-pytorch/) | PyTorch | ~10–15 min |

Launch using any GPU-enabled kernel of the listed framework that's
available to you — this guide doesn't assume a particular GPU
size/allocation; ask your instructor which one to pick if you're not
sure. Time estimates are rough and will vary with your GPU allocation.
Modules are independent — assign any subset in any order.

Four pairs of modules solve the *same* task in both frameworks, so
students can compare them directly: **01 ↔ 07** (MLP on MNIST), **02 ↔ 03**
(CNN on CIFAR-10), **04 ↔ 08** (transfer learning), and **05 ↔ 09** (NLP
text classification). Module 06 (reinforcement learning) has no
TensorFlow counterpart.

---

## Which environment should I use?

All three run identical code and produce the same results — pick based on
what your course uses, or whatever you're comfortable with.

| Environment | Open | Best for |
|---|---|---|
| **JupyterLab** | `notebook.ipynb` | Interactive, cell-by-cell exploration with plots shown inline |
| **VS Code** (browser, code-server) | `script.py` | A full code editor — file explorer, syntax highlighting, integrated terminal |
| **Terminal** (browser, no editor) | `script.py` | The leanest option — just run it and check the printed output/saved plots |

VS Code and Terminal are two separate buttons on the platform (a kernel
may have one, both, or neither enabled by your admin) — see their
step-by-step sections below.

---

## Step-by-step: JupyterLab

Do this once per module, right after you spawn your session.

**1. Launch JupyterLab from the portal.**
When you spawn or open your notebook server, pick a kernel of the
framework listed in the table above for the module you're doing (see
[03-launching-jupyterlab.md](../../user-guide/03-launching-jupyterlab.md)
if you're not sure how). If you already have a session running with the
wrong kernel, open a notebook and switch kernels from the kernel selector
in the top-right corner — you don't need to restart the whole session.

**2. Open a Terminal inside JupyterLab.**
`File` → `New` → `Terminal`.

**3. Copy-paste this, once, into that terminal:**

```bash
cd ~/
git clone https://github.com/vitu-mafeni/ml-platform.git ml-platform
cd ml-platform/docs/demos/student-examples
ls
```

You should see the nine numbered module folders. Your home folder
(`/home/jovyan/`) is persistent (see
[07-storage.md](../../user-guide/07-storage.md)), so you only need to do
this clone once — it will still be there next time you log in.

> If `git clone` fails with a permission/authentication error, this repo is
> private — ask your instructor for either a personal access token to use
> in the URL, or a zip export you can upload instead via the JupyterLab
> file browser (drag-and-drop into the file panel, then
> `unzip ml-platform.zip` in the terminal).

**4. Open the module notebook.**
In the JupyterLab file browser (left panel), navigate into
`ml-platform/docs/demos/student-examples/<module-folder>/` and
double-click `notebook.ipynb`.

**5. Confirm the kernel, then run all cells.**
Check the kernel name shown in the top-right corner of the notebook
matches the module's framework (PyTorch or TensorFlow) from the table
above. If it doesn't, click it and pick the right one from the dropdown.
Then `Run` → `Run All Cells`, or step through cell by cell with
`Shift+Enter`.

**6. Pull the latest version later (optional).**
If your instructor updates the examples, get the update with:

```bash
cd ~/ml-platform
git pull
```

---

## Step-by-step: VS Code or Terminal

This platform's "VS Code" and "Terminal" are browser-based tools
(`code-server` and a browser terminal) attached to your running GPU
kernel — not your local machine connecting in over SSH. Each is only
available on kernel profiles your admin has enabled it for; a kernel may
offer one, both, or neither. If you don't see the button you expect, ask
your instructor, or just use JupyterLab instead.

> ### ⚠️ Read this before you start: storage works differently here
>
> JupyterLab's home folder (`/home/jovyan/`) is persistent. **VS Code and
> Terminal both run on the kernel's own *ephemeral* disk instead** — files
> you create there, including anything you `git clone`, are gone the
> moment the kernel stops, restarts, or gets reclaimed for idling. There
> is no autosave and no grace period.
>
> The fix: click **"Sync to Home"** on that kernel's row in **My
> Workloads** *before* you stop it, restart it, or step away for a while.
> That copies `/home/jovyan` (the same path both tools open into) from the
> kernel's disk into your real, persistent home folder. Do this every
> time you're done for the session — it is a manual, one-click action,
> not automatic.

### Using VS Code

**1. Launch a VS Code-enabled kernel from the portal.**
Spawn or open a kernel of the module's framework, then click its
**"VS Code ↗"** button from **My Workloads**. It opens in a new browser
tab.

**2. Open a terminal inside VS Code.**
`Terminal` → `New Terminal`.

**3. Copy-paste this, once, into that terminal:**

```bash
cd ~/
git clone https://github.com/vitu-mafeni/ml-platform.git ml-platform
cd ml-platform/docs/demos/student-examples
ls
```

> Same fallback as JupyterLab if `git clone` fails: ask your instructor
> for a token, or upload a zip via VS Code's file explorer and
> `unzip ml-platform.zip`.

**4. Run a module's script:**

```bash
cd 01-mlp-mnist-pytorch
python script.py
```

(swap the folder name for whichever module you're doing — you're already
inside `ml-platform/docs/demos/student-examples` from step 3). It runs
top to bottom and prints progress the same way the notebook does. Plots
are saved as PNG files under `results/` in that folder — open them from
VS Code's file explorer — since a terminal can't show inline plots the
way JupyterLab can.

If your VS Code has the Python extension with cell support, you can also
open `script.py` directly and run it cell-by-cell: each `# %%` line marks
a cell, the same way notebook cells work.

**5. Before you stop this kernel: Sync to Home.**
Go back to **My Workloads** and click **Sync to Home** on this kernel.
Skipping this step means losing everything you just did — see the
warning above.

### Using Terminal only

The leanest option — a plain browser shell, no code editor. Good if you
just want to run a module and read its output/plots.

**1. Launch a Terminal-enabled kernel from the portal.**
Spawn or open a kernel of the module's framework, then click its
**"Terminal ↗"** button from **My Workloads**. It opens in a new browser
tab, straight into a shell.

**2. Copy-paste this, once:**

```bash
cd ~/
git clone https://github.com/vitu-mafeni/ml-platform.git ml-platform
cd ml-platform/docs/demos/student-examples
ls
```

**3. Run a module's script:**

```bash
cd 01-mlp-mnist-pytorch
python script.py
```

There's no file browser here, so read the plots with, e.g.,
`ls results/` to confirm they were saved — you'll need JupyterLab or VS
Code (or `scp`/download tooling your admin provides) to actually view a
PNG from a plain terminal. If you want to glance at the script itself,
`cat script.py` or `less script.py` work without needing an editor.

**4. Before you stop this kernel: Sync to Home.**
Go back to **My Workloads** and click **Sync to Home** on this kernel —
same warning as VS Code above, this terminal's disk is just as ephemeral.

---

## If something goes wrong

| Problem | What to do |
|---|---|
| The GPU-check step raises an `AssertionError` | You're on the wrong kernel (or a CPU-only one). Go to the portal's **Servers** tab, stop the session, spawn a new one on a GPU kernel of the right framework, then try again. |
| `git clone` hangs or times out | Your cluster may not have outbound internet access to GitHub — ask your admin, or use the zip-upload fallback above. |
| A dataset download hangs or fails | Same as above — datasets (MNIST, CIFAR-10, AG News) download from the internet on first run. Ask your admin whether the cluster allows this, or whether a pre-cached dataset mirror exists. |
| `%pip install` / `pip install` step fails | Check your internet access the same way; if packages are mirrored internally, ask your admin for the correct index URL to add. |
| Kernel dropdown doesn't show a GPU kernel for the framework you need | Your account may not have been granted that kernel — see [06-roles-and-permissions.md](../../user-guide/06-roles-and-permissions.md) and ask your admin. |
| No "VS Code ↗" or "Terminal ↗" button on your kernel | That kernel profile doesn't have it enabled — ask your instructor/admin, or use JupyterLab and `notebook.ipynb` instead. |
| Your `git clone` / work "disappeared" in VS Code or Terminal | You didn't click **Sync to Home** before the kernel stopped or restarted — see the storage warning above. It's not recoverable; you'll need to clone again. |

---

## For instructors

- Each module's `README.md` documents its objective, framework, and
  grading-friendly exercises built into the material itself (a numbered
  "Exercises" section with an empty scratch cell/section at the end).
- No manual save/resume or checkpoint logic is used here — these are
  finite training runs (fixed epoch counts), unlike the platform's
  checkpoint/restore product demos alongside this folder in
  [`docs/demos/`](../README.md), which are a different, unrelated set of
  materials for showing off the platform itself rather than for
  coursework.
- VS Code and Terminal only work on kernel profiles an admin built with
  those features enabled (`kernel_vscode_enabled` /
  `kernel_terminal_enabled`, independently of each other) — confirm which
  of your course's kernel profiles have which before telling students to
  expect a given button. See
  [12-gpu-profiles.md](../../admin-guide/12-gpu-profiles.md) /
  [04-kernel-profile-management.md](../../admin-guide/04-kernel-profile-management.md).
- If you want students cloning *only* this coursework folder (not this
  whole platform repository), split it into its own repo once, e.g.:
  ```bash
  git subtree split --prefix=docs/demos/student-examples -b student-examples-only
  ```
  then push that branch to a new, separate GitHub repo and swap the clone
  URL in this README accordingly.
