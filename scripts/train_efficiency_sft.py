"""Continue an existing G3 LoRA on reviewed, matched efficiency demonstrations.

Public reference entry point, extracted from the completed experiment's method.
Use the original merged SFT model as --model, NOT a model already merged with G3.
This entry point starts a new run; it does not replay the private control system.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from tooltune.efficiency import AccumulationCheck, encode_reviewed_rows, validate_pairs
from tooltune.io import load_config, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local original merged SFT base")
    parser.add_argument(
        "--adapter", required=True, help="Local original G3 LoRA directory"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--paired-data", type=Path, required=True)
    parser.add_argument("--arm", choices=["low_call", "correct_random"], required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; never overwritten"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/efficiency_sft.json")
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, paired = read_jsonl(args.data), read_jsonl(args.paired_data)
    validate_pairs(rows, paired, cfg["matched_rows"], cfg["replay_rows"])
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run each arm independently in a single process")
    if args.output.exists():
        raise FileExistsError(
            "Choose a new output directory; partial runs are preserved"
        )

    import torch
    from datasets import Dataset
    from peft import PeftModel, get_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    if torch.cuda.device_count() != 1:
        raise ValueError(
            "Select one visible GPU for this arm with CUDA_VISIBLE_DEVICES"
        )
    set_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = encode_reviewed_rows(tokenizer, rows, cfg["max_length"])
    # Check the control's arrays as well, without sampling, judging or training it.
    encode_reviewed_rows(tokenizer, paired, cfg["max_length"])
    if tokenizer.pad_token_id is None:
        raise ValueError("The original tokenizer must define its padding token")
    args.output.mkdir(parents=True, exist_ok=False)
    run = {"arm": args.arm, "config": cfg, "model": args.model, "adapter": args.adapter}
    run["data_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    run["paired_data_sha256"] = hashlib.sha256(
        args.paired_data.read_bytes()
    ).hexdigest()
    (args.output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, args.adapter, is_trainable=True, local_files_only=True
    )
    original = load_peft_weights(args.adapter, device="cpu")
    loaded = get_peft_model_state_dict(model)
    if set(original) != set(loaded) or any(
        not torch.equal(original[key], loaded[key].detach().cpu()) for key in original
    ):
        raise RuntimeError("Existing adapter was not loaded exactly")
    if not any(p.requires_grad for p in model.parameters()) or any(
        "lora_" not in name for name, p in model.named_parameters() if p.requires_grad
    ):
        raise RuntimeError("Only the existing LoRA parameters may be trained")
    del original, loaded
    check = AccumulationCheck(
        len(rows), cfg["epochs"], cfg["gradient_accumulation_steps"]
    )

    def collate(examples):
        length = math.ceil(max(len(row["input_ids"]) for row in examples) / 8) * 8
        return {
            key: torch.tensor(
                [row[key] + [pad] * (length - len(row[key])) for row in examples]
            )
            for key, pad in (
                ("input_ids", tokenizer.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            )
        }

    class CheckedTrainer(SFTTrainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            if (
                not self.model_accepts_loss_kwargs
                or self.accelerator.gradient_accumulation_steps != 1
            ):
                raise RuntimeError(
                    "Unexpected TRL loss scaling; use the pinned training dependencies"
                )
            denominator = (
                None if num_items_in_batch is None else int(num_items_in_batch)
            )
            check.add(
                int(inputs["labels"].ne(-100).sum()),
                denominator,
                self.current_gradient_accumulation_steps,
            )
            # Keep TRL's native sum-CE / group-token-count path. Do not divide
            # this loss again by 16 (or by the 10 examples in a tail group).
            loss = super().training_step(
                model, inputs, num_items_in_batch=num_items_in_batch
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            return loss

    class CheckSteps(TrainerCallback):
        def on_train_begin(self, training_args, state, control, **kwargs):
            if state.global_step != 0 or state.max_steps != check.total_updates:
                raise RuntimeError(
                    "Unexpected starting step or number of optimizer updates"
                )

        def on_step_end(self, training_args, state, control, **kwargs):
            record = check.finish(state.global_step)
            with (args.output / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if (
                logs
                and "grad_norm" in logs
                and not math.isfinite(float(logs["grad_norm"]))
            ):
                raise RuntimeError("Nonfinite gradient norm")

    trainer = CheckedTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(args.output),
            bf16=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            warmup_ratio=cfg["warmup_ratio"],
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=cfg["max_length"],
            max_grad_norm=1.0,
            optim="adamw_torch_fused",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            weight_decay=0.0,
            loss_type="nll",
            packing=False,
            group_by_length=False,
            dataloader_drop_last=False,
            dataloader_num_workers=0,
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            save_strategy="steps",
            save_steps=check.updates_per_epoch,
            save_total_limit=None,
            save_only_model=False,
            logging_steps=1,
            report_to="none",
            seed=cfg["seed"],
            data_seed=cfg["seed"],
        ),
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(encoded),
        data_collator=collate,
        callbacks=[CheckSteps()],
    )
    if (
        trainer.compute_loss_func is not None
        or trainer.label_smoother is not None
        or trainer.args.use_liger_kernel
    ):
        raise RuntimeError("Require the original native NLL loss without extra scaling")
    trainer.train()
    if check.step != check.total_updates or check.pending:
        raise RuntimeError("Training ended before all configured updates completed")
    trainer.save_model(str(args.output / "last-adapter"))


if __name__ == "__main__":
    main()
