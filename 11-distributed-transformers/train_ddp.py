"""
Causal-LM fine-tuning with plain PyTorch DDP + Hugging Face transformers.

Works unchanged with:
    python train_ddp.py ...                      # self-spawns one rank per visible GPU
    torchrun --nproc_per_node=N train_ddp.py ... # standard launcher
    python -m torch.distributed.run ...          # same thing as torchrun
"""
import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_steps", type=int, default=-1, help="stop early (smoke tests)")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--output_dir", default="outputs/gpt2-ddp")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="force CPU + gloo (logic testing)")
    p.add_argument("--nproc", type=int, default=0,
                   help="ranks when launched with plain `python` (default: #GPUs, or 2 on CPU)")
    return p.parse_args()


def build_blocks(args, tokenizer):
    raw = load_dataset(args.dataset, args.dataset_config)

    def tok(batch):
        return tokenizer(batch["text"])

    def group(batch):
        ids = [t for seq in batch["input_ids"] for t in seq]
        n = (len(ids) // args.block_size) * args.block_size
        return {"input_ids": [ids[i:i + args.block_size] for i in range(0, n, args.block_size)]}

    cols = raw["train"].column_names
    tokenized = raw.map(tok, batched=True, remove_columns=cols, desc="tokenize")
    return tokenized.map(group, batched=True, remove_columns=tokenized["train"].column_names, desc="group")


def collate(batch):
    ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    return {"input_ids": ids, "labels": ids.clone()}  # HF shifts labels internally


@torch.no_grad()
def evaluate(model, loader, device, amp_ctx):
    model.eval()
    total = torch.zeros(2, device=device)  # [sum of loss*tokens, tokens]
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp_ctx():
            loss = model(**batch).loss
        n = batch["input_ids"].numel()
        total += torch.tensor([loss.item() * n, n], device=device)
    dist.all_reduce(total)  # sum across ranks
    model.train()
    return math.exp(total[0] / total[1])


def run(rank, local_rank, world_size, args):
    use_cuda = torch.cuda.is_available() and not args.cpu
    backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.manual_seed(args.seed)  # same seed => identical init on all ranks
    is_main = rank == 0

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # rank 0 downloads/tokenizes first, the others then hit the HF cache
    if not is_main:
        dist.barrier()
    data = build_blocks(args, tokenizer)
    if is_main:
        dist.barrier()

    train_sampler = DistributedSampler(data["train"], world_size, rank, shuffle=True, seed=args.seed)
    train_loader = DataLoader(data["train"], batch_size=args.per_device_batch_size,
                              sampler=train_sampler, collate_fn=collate,
                              num_workers=2, pin_memory=use_cuda, drop_last=True)
    val_sampler = DistributedSampler(data["validation"], world_size, rank, shuffle=False)
    val_loader = DataLoader(data["validation"], batch_size=args.per_device_batch_size,
                            sampler=val_sampler, collate_fn=collate)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)
    model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    no_decay = ("bias", "ln_", "layernorm", "norm")
    decay_p = [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n.lower() for k in no_decay)]
    other_p = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n.lower() for k in no_decay)]
    optim = torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": args.weight_decay}, {"params": other_p, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=use_cuda)

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(optim, int(args.warmup_ratio * total_steps), total_steps)

    amp_ctx = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if args.bf16 else nullcontext
    if is_main:
        eff = args.per_device_batch_size * args.grad_accum * world_size
        print(f"world_size={world_size} backend={backend} effective_batch={eff} "
              f"steps/epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    model.train()
    step, t0, running = 0, time.time(), torch.zeros(1, device=device)
    done = False
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # reshuffle differently each epoch
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            boundary = (i + 1) % args.grad_accum == 0
            # skip the gradient all-reduce on non-boundary micro-steps
            with (nullcontext() if boundary else model.no_sync()):
                with amp_ctx():
                    loss = model(**batch).loss / args.grad_accum
                loss.backward()
            running += loss.detach()
            if not boundary:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                dist.all_reduce(running, op=dist.ReduceOp.AVG)
                if is_main:
                    sec = (time.time() - t0) / args.log_every
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"loss {running.item() / args.log_every:.4f} lr {sched.get_last_lr()[0]:.2e} "
                          f"{sec:.2f}s/step", flush=True)
                running.zero_()
                t0 = time.time()
            if step >= total_steps:
                done = True
                break
        ppl = evaluate(model, val_loader, device, amp_ctx)
        if is_main:
            print(f"== epoch {epoch} val_ppl {ppl:.2f}", flush=True)
        if done:
            break

    if is_main:  # save once, from the unwrapped model
        os.makedirs(args.output_dir, exist_ok=True)
        model.module.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"saved to {args.output_dir}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def _spawn_entry(rank, world_size, args):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=os.environ.get("MASTER_PORT", "29500"))
    run(rank, rank, world_size, args)


if __name__ == "__main__":
    args = parse_args()
    if "RANK" in os.environ:  # launched by torchrun / deepspeed / accelerate
        run(int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), args)
    else:                     # plain `python train_ddp.py`
        n = args.nproc or (torch.cuda.device_count() if torch.cuda.is_available() and not args.cpu else 2)
        n = max(n, 1)
        mp.spawn(_spawn_entry, args=(n, args), nprocs=n, join=True)
