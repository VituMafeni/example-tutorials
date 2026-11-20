"""
Causal-LM fine-tuning with the Hugging Face Trainer.

The Trainer reads RANK / LOCAL_RANK / WORLD_SIZE from the environment, so the same
file runs under python, torchrun, deepspeed and accelerate with no code changes.
DeepSpeed is switched on purely from the CLI:  --deepspeed ds_zero2.json
"""
import os
from dataclasses import dataclass

from datasets import DownloadConfig, load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          HfArgumentParser, Trainer, TrainingArguments, set_seed)


@dataclass
class DataArgs:
    model_name: str = "gpt2"
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    block_size: int = 512
    offline: bool = False


def main():
    data_args, targs = HfArgumentParser((DataArgs, TrainingArguments)).parse_args_into_dataclasses()
    set_seed(targs.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        data_args.model_name, local_files_only=data_args.offline
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config,
        download_config=DownloadConfig(local_files_only=data_args.offline),
    )
    bs = data_args.block_size

    def tok(batch):
        return tokenizer(batch["text"])

    def group(batch):
        ids = [t for seq in batch["input_ids"] for t in seq]
        n = (len(ids) // bs) * bs
        return {"input_ids": [ids[i:i + bs] for i in range(0, n, bs)]}

    # rank 0 builds the cache, other ranks reuse it
    with targs.main_process_first(desc="tokenize + group"):
        cols = raw["train"].column_names
        tokenized = raw.map(tok, batched=True, remove_columns=cols)
        lm = tokenized.map(group, batched=True, remove_columns=tokenized["train"].column_names)

    model = AutoModelForCausalLM.from_pretrained(
        data_args.model_name, local_files_only=data_args.offline
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=lm["train"],
        eval_dataset=lm["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    ckpt = None
    if os.path.isdir(targs.output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        ckpt = get_last_checkpoint(targs.output_dir)  # auto-resume after a crash / pod restart
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model()  # rank-0 only; handles ZeRO-3 weight gathering
    tokenizer.save_pretrained(targs.output_dir)
    metrics = trainer.evaluate()
    if trainer.is_world_process_zero():
        print("eval:", metrics)


if __name__ == "__main__":
    main()
