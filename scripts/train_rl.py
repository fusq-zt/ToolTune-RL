"""GRPO training with tool-aware rollouts and entropy-guided branching."""

import argparse
from collections import deque
import json
from pathlib import Path
from tooltune.io import load_config, model_identity, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Local merged SFT model (fixed reference)"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--method", choices=["G0", "G1", "G2", "G3", "G4"], default="G3"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    import inspect
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
        set_seed,
    )
    import trl
    from trl import GRPOConfig, GRPOTrainer
    from tooltune.endpoints import ColocatedEndpoint
    from tooltune.evaluation import evaluate_checkpoint, is_better
    from tooltune.gradient_checks import check_loss_contract
    from tooltune.memory import install_long_context_offload
    from tooltune.protocol import messages_for, TOOLS
    from tooltune.rewards import group_rewards, efficiency_weight
    from tooltune.rollout import Rollout
    from tooltune.sandbox import preflight
    from tooltune.sync_guard import install_sync_guard

    if (
        trl.__version__ != "0.27.2"
        or "ToolTune: external rollout environment mask"
        not in Path(inspect.getfile(GRPOTrainer)).read_text()
    ):
        raise RuntimeError(
            "Install the pinned training dependencies and run scripts/install_trl_adapter.py"
        )
    preflight()
    cfg = load_config(args.config)
    rl = cfg["rl"]
    group_size = rl["num_generations"]
    method = cfg["methods"][args.method]
    if group_size != 4:
        raise ValueError("The released experiment configuration uses groups of four")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new output directory or --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = model_identity(args.model)
    tasks, dev = read_jsonl(args.data), read_jsonl(args.dev)
    if (
        not tasks
        or not dev
        or {r["task_id"] for r in tasks} & {r["task_id"] for r in dev}
    ):
        raise ValueError(
            "Training and development inputs must be nonempty and disjoint"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, padding_side="left"
    )
    lookup = {
        tokenizer.apply_chat_template(
            messages_for(t), tools=TOOLS, tokenize=False, add_generation_prompt=True
        ): t
        for t in tasks
    }
    if len(lookup) != len(tasks):
        raise ValueError("Duplicate training prompt")
    engine = Rollout(tokenizer, None, seed=cfg["seed"], **cfg["rollout"])
    if args.calibration:
        calibration = load_config(args.calibration)
        if calibration["model_identity"] != identity:
            raise ValueError("Entropy calibration must use this exact SFT reference")
        for key, values in calibration["values"].items():
            engine.history[key] = deque(values, maxlen=2048)
    if (
        method["mode"] in ("entropy", "calibrated")
        and not args.calibration
        and not args.resume
    ):
        raise ValueError(
            "Initialize entropy statistics with --calibration before G3/G4"
        )
    if args.resume:
        saved = load_config(args.resume / "rollout_state.json")
        if (
            saved["model_identity"] != identity
            or saved["method"] != args.method
            or saved["config"] != cfg
        ):
            raise ValueError(
                "Resume requires the same base model, method and configuration"
            )
        engine.batch_number = saved["batch_number"]
        for key, values in saved["calibration"].items():
            engine.history[key] = deque(values, maxlen=2048)

    def rollout_func(prompts, trainer):
        assert len(prompts) % group_size == 0
        assert all(
            prompts[i : i + group_size] == [prompts[i]] * group_size
            for i in range(0, len(prompts), group_size)
        )
        selected = [lookup[p] for p in prompts[::group_size]]
        if engine.endpoint is None:
            engine.endpoint = ColocatedEndpoint(trainer, args.output)
        engine.endpoint.begin()
        states = engine.sample(
            selected, group_size=group_size, mode=method["mode"], model=trainer.model
        )
        engine.endpoint.finish(states)
        records = [engine.record(s) for s in states]
        assert len(states) == len(prompts)
        with (args.output / "rollouts.jsonl").open("a", encoding="utf-8") as stream:
            for row in records:
                stream.write(
                    json.dumps(dict(row, policy_step=trainer.state.global_step)) + "\n"
                )
        return {
            "prompt_ids": [s.prompt for s in states],
            "completion_ids": [s.ids for s in states],
            "logprobs": [s.logprobs for s in states],
            "env_mask": [s.mask for s in states],
            "rollout_records": records,
        }

    def reward(prompts, completions, rollout_records, trainer_state, **kwargs):
        assert len(prompts) == len(completions) == len(rollout_records)
        return group_rewards(
            rollout_records,
            group_size=group_size,
            lam=efficiency_weight(trainer_state.global_step, trainer_state.max_steps),
            efficiency=method["efficiency"],
        )

    class SaveAndSelect(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            checkpoint = args.output / f"checkpoint-{state.global_step}"
            payload = {
                "model_identity": identity,
                "method": args.method,
                "config": cfg,
                "batch_number": engine.batch_number,
                "calibration": {k: list(v) for k, v in engine.history.items()},
            }
            (checkpoint / "rollout_state.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            engine.endpoint.sync_for_evaluation()
            try:
                report = evaluate_checkpoint(
                    tokenizer,
                    engine.endpoint,
                    dev,
                    args.output / "dev" / f"step-{state.global_step}",
                )
            finally:
                engine.endpoint.sleep()
            report.update(step=state.global_step, path=str(checkpoint))
            best_path = args.output / "selected-checkpoint.json"
            best = load_config(best_path) if best_path.exists() else None
            if is_better(report, best):
                best_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    set_seed(cfg["seed"])
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.config.use_cache = False
    config = GRPOConfig(
        output_dir=str(args.output),
        bf16=True,
        per_device_train_batch_size=1,
        max_steps=rl["updates"],
        gradient_accumulation_steps=rl["gradient_accumulation_steps"],
        generation_batch_size=rl["generation_batch_size"],
        num_generations=group_size,
        learning_rate=rl["learning_rate"],
        warmup_ratio=rl["warmup_ratio"],
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_prompt_length=4096,
        max_completion_length=cfg["rollout"]["model_tokens"],
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.45,
        vllm_enable_sleep_mode=True,
        vllm_max_model_length=cfg["rollout"]["total_tokens"],
        temperature=cfg["rollout"]["temperature"],
        top_p=1.0,
        top_k=-1,
        beta=rl["beta"],
        epsilon=rl["epsilon"],
        scale_rewards=rl["scale_rewards"],
        loss_type=rl["loss_type"],
        vllm_importance_sampling_correction=rl["vllm_importance_sampling_correction"],
        vllm_importance_sampling_mode=rl["vllm_importance_sampling_mode"],
        vllm_importance_sampling_cap=rl["vllm_importance_sampling_cap"],
        num_iterations=1,
        disable_dropout=True,
        mask_truncated_completions=False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=rl["save_steps"],
        report_to="none",
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward,
        rollout_func=rollout_func,
        train_dataset=Dataset.from_list([{"prompt": p} for p in lookup]),
        processing_class=tokenizer,
        peft_config=LoraConfig(**cfg["lora"]),
        callbacks=[SaveAndSelect()],
    )
    if trainer.accelerator.num_processes != 1:
        raise ValueError("Launch one process for this training entry point")
    check_loss_contract(trainer)
    install_sync_guard(trainer, tokenizer, args.output)
    install_long_context_offload(trainer, args.output)
    (args.output / "config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    try:
        trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        trainer.save_model(str(args.output / "final-adapter"))
    finally:
        engine.pool.shutdown()


if __name__ == "__main__":
    main()
