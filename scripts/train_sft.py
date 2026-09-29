"""Train a role-masked LoRA SFT policy with the official TRL SFTTrainer."""

import argparse
import math
import os
from pathlib import Path
from tooltune.io import load_config, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--resume")
    args = parser.parse_args()
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    cfg = load_config(args.config)
    sft = cfg["sft"]
    rows = read_jsonl(args.data)
    if not rows or len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty, unique SFT examples")
    for row in rows:
        ids, labels, mask = (row[k] for k in ("input_ids", "labels", "attention_mask"))
        assert 0 < len(ids) <= sft["max_length"] and len(ids) == len(labels) == len(
            mask
        )
        assert any(label != -100 for label in labels)
        assert all(label in (-100, token) for label, token in zip(labels, ids))
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(
            "Use a new output directory or an explicit --resume checkpoint"
        )
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if sft["effective_batch_size"] % world:
        raise ValueError("Effective batch size must be divisible by the process count")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    def collate(examples):
        length = math.ceil(max(len(e["input_ids"]) for e in examples) / 8) * 8
        return {
            key: torch.tensor(
                [e[key] + [pad] * (length - len(e[key])) for e in examples]
            )
            for key, pad in [
                ("input_ids", tokenizer.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            ]
        }

    set_seed(cfg["seed"])
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.config.use_cache = False
    config = SFTConfig(
        output_dir=str(args.output),
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=sft["effective_batch_size"] // world,
        num_train_epochs=sft["epochs"],
        learning_rate=sft["learning_rate"],
        warmup_ratio=sft["warmup_ratio"],
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=sft["max_length"],
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        save_steps=max(
            1, math.ceil(math.ceil(len(rows) / sft["effective_batch_size"]) / 4)
        ),
        save_strategy="steps",
        save_total_limit=5,
        logging_steps=1,
        report_to="none",
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(
            [{k: r[k] for k in ("input_ids", "attention_mask", "labels")} for r in rows]
        ),
        data_collator=collate,
        peft_config=LoraConfig(**cfg["lora"]),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(args.output / "last-adapter"))


if __name__ == "__main__":
    main()
