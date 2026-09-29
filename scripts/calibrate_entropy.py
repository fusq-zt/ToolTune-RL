"""Initialize common entropy histories using fixed training questions, without rewards.

Sampling and entropy measurement use separate subprocesses so their model copies
do not need to occupy accelerator memory simultaneously.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tooltune.io import read_jsonl, model_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=["all", "sample", "measure"],
        default="all",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    traces = args.output.with_suffix(".trajectories.jsonl")
    if args.output.exists():
        raise FileExistsError("Use a new calibration output")
    if args.phase == "all":
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model",
            args.model,
            "--data",
            str(args.data),
            "--output",
            str(args.output),
        ]
        subprocess.run(command + ["--phase", "sample"], check=True)
        subprocess.run(command + ["--phase", "measure"], check=True)
        return
    from transformers import AutoTokenizer
    from tooltune.rollout import Rollout, PathState

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.phase == "sample":
        from tooltune.endpoints import InferenceEndpoint
        from tooltune.sandbox import preflight

        if traces.exists():
            raise FileExistsError(traces)
        tasks = read_jsonl(args.data)
        if not tasks or any(t.get("split") != "rl" for t in tasks):
            raise ValueError(
                "Calibration accepts only the prepared RL training partition"
            )
        tasks = sorted(
            tasks,
            key=lambda t: hashlib.sha256(
                ("entropy:" + t["task_id"]).encode()
            ).hexdigest(),
        )[:128]
        preflight()
        engine = Rollout(tokenizer, InferenceEndpoint(args.model), seed=13031)
        traces.parent.mkdir(parents=True, exist_ok=True)
        try:
            with traces.open("x", encoding="utf-8") as stream:
                for start in range(0, len(tasks), 8):
                    for state in engine.sample(
                        tasks[start : start + 8], group_size=2, mode="full"
                    ):
                        stream.write(json.dumps(asdict(state)) + "\n")
        finally:
            engine.pool.shutdown()
        return
    import torch
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    engine = Rollout(tokenizer, None, seed=13031)
    values = defaultdict(list)
    task_ids = set()
    try:
        for row in read_jsonl(traces):
            state = PathState(**row)
            task_ids.add(state.task["task_id"])
            for observation in state.observations:
                delta = engine.entropy(model, state, observation)
                if delta is not None:
                    values["global"].append(delta)
                    values[observation["tool"]].append(delta)
    finally:
        engine.pool.shutdown()
    if len(values["global"]) < 32:
        raise RuntimeError(
            "Insufficient entropy observations; inspect sampled training trajectories"
        )
    report = {
        "model_identity": model_identity(args.model),
        "task_ids": sorted(task_ids),
        "values": dict(values),
        "selection": "fixed training IDs; no rewards used",
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("Entropy calibration saved")


if __name__ == "__main__":
    main()
