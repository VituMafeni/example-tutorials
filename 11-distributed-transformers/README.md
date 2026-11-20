# Demo 6 — Distributed Transformer Training (DDP / torchrun / DeepSpeed / accelerate)

A reference for ML engineers: fine-tune a Hugging Face causal LM (`gpt2`)
with `torch.distributed`, launched four different ways. Unlike demos 1–4
this is **not** a checkpoint/restore demo — see the caveats below.

## Files

| File | What it is |
|---|---|
| `train_ddp.py` | Plain PyTorch DDP + transformers: `DistributedSampler`, grad accumulation with `no_sync`, bf16 autocast, rank-0 save, all-reduced perplexity. Runs under `python` (self-spawns ranks) or `torchrun`. |
| `train_hf.py` | HF `Trainer` script. One file for `python`, `torchrun`, `deepspeed` and `accelerate`; DeepSpeed is enabled from the CLI with `--deepspeed`. Auto-resumes from the last Trainer checkpoint. |
| `ds_zero2.json` | ZeRO-2 config. `torch_adam: true` avoids JIT-compiling FusedAdam (the image has no `nvcc`). |
| `ds_zero3_offload.json` | ZeRO-3 with optimizer + parameter CPU offload, for fitting larger models in a small GPU slice. |

## What the platform allows (read first)

- **One GPU per session.** A session is a single kernel pod with one HAMi
  vGPU slice (`hami.io/vgpu-devices: 1`; 6 GB on the quarter profile), so a
  real NCCL run has world size 1. Several NCCL ranks on one GPU fail with
  "Duplicate GPU detected". To exercise multi-rank logic, use
  `--cpu --nproc 2` (gloo).
- **No multi-node yet.** `PyTorchJob` is on the roadmap
  ([12-future-architecture.md](../../technical/12-future-architecture.md)).
  The multi-node command below is for a cluster that has it.
- **Not CRIU-safe.** The platform's checkpoint/restore cannot represent NCCL
  state (see the [demos README](../README.md)). Use the Trainer's own
  checkpoints (`--save_strategy`) for distributed runs.

## Setup

Use a PyTorch + transformers GPU profile (e.g. the
`vitu1/kernel-huggingface-gpu` image from
[testing-custom-images-with-real-ml-images.md](../../requirements/testing-custom-images-with-real-ml-images.md))
with the **Terminal** or **VS Code** option enabled, then run the commands
below there (or prefix them with `!` in a notebook cell).

```bash
pip install deepspeed   # only needed for section 3; not in the image
```

The scripts download `gpt2` and `Salesforce/wikitext` from the Hugging Face
Hub, so the pod needs outbound internet access.

Shared Trainer arguments, to keep the commands short:

```bash
ARGS="--model_name gpt2 --output_dir outputs/gpt2-hf --bf16 --num_train_epochs 1 \
--per_device_train_batch_size 8 --gradient_accumulation_steps 2 --learning_rate 5e-5 \
--lr_scheduler_type cosine --warmup_ratio 0.03 --eval_strategy epoch --save_strategy epoch \
--save_total_limit 2 --logging_steps 10 --report_to none"
```

## 1. python

```bash
python train_ddp.py --bf16                      # one rank per visible GPU (1 on the platform)
python train_ddp.py --cpu --nproc 2 --max_steps 20   # 2-rank logic test with gloo
python train_hf.py $ARGS                        # Trainer, single process
```

## 2. torchrun

```bash
torchrun --standalone --nproc_per_node=gpu train_ddp.py --bf16
torchrun --standalone --nproc_per_node=gpu train_hf.py $ARGS
python -m torch.distributed.run --standalone --nproc_per_node=gpu train_hf.py $ARGS   # same thing
```

Multi-node, once the platform has `PyTorchJob` (run on each node with its own
`NODE_RANK`; the scripts are unchanged):

```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$NODE_RANK \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 train_hf.py $ARGS
```

## 3. deepspeed

```bash
deepspeed --num_gpus=1 train_hf.py $ARGS --deepspeed ds_zero2.json
deepspeed --num_gpus=1 train_hf.py $ARGS --deepspeed ds_zero3_offload.json --per_device_train_batch_size 2
```

`torchrun --nproc_per_node=N train_hf.py $ARGS --deepspeed ds_zero2.json` is
equivalent to `deepspeed --num_gpus=N`.

## 4. accelerate

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 train_hf.py $ARGS
accelerate launch --use_deepspeed --zero_stage 2 --num_processes 1 --mixed_precision bf16 train_hf.py $ARGS
```

Don't also pass `--deepspeed` with `--use_deepspeed`; accelerate generates the
DeepSpeed settings itself.

## Tuning for a small slice

- CUDA OOM (HAMi enforces the hard cap): lower `--per_device_batch_size`,
  add `--grad_checkpointing` (`train_ddp.py`) or `--gradient_checkpointing true`
  (`train_hf.py`), or switch to `ds_zero3_offload.json`.
- DataLoader "bus error" with a small `/dev/shm`: set `num_workers=0` in
  `train_ddp.py`.

## Verification status

Tested on CPU (gloo, 2 ranks, tiny GPT-2): `train_ddp.py` under `python` and
`torchrun`; `train_hf.py` under `python` and `torchrun`. `accelerate launch
--cpu` runs to completion but multi-rank startup was not confirmed.
**Not yet run:** NCCL on a GPU, either DeepSpeed config, or anything on the
platform itself.
