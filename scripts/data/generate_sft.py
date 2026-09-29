"""Bounded same-4B rejection sampling, two candidates per SFT-only question."""

import argparse
import json
from pathlib import Path
import time
from transformers import AutoTokenizer
from tooltune.protocol import encode_sft
from tooltune.rollout import Rollout
from tooltune.endpoints import InferenceEndpoint
from tooltune.sandbox import preflight

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--shards", type=int, default=2)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--round", type=int, choices=[0, 1], default=0)
args = parser.parse_args()
preflight()
out = args.root / "data/sft"
out.mkdir(parents=True, exist_ok=True)
suffix = f"{args.shard}" if args.round == 0 else f"{args.shard}-r1"
accepted_path = out / f"generated-{suffix}.jsonl"
traces = out / f"candidates-{suffix}.jsonl"
completed_path = out / f"completed-{suffix}.jsonl"
public = {
    r["task_id"]
    for r in map(json.loads, (out / "public.jsonl").read_text().splitlines())
}
completed = (
    {r["task_id"] for r in map(json.loads, completed_path.read_text().splitlines())}
    if completed_path.exists()
    else set()
)
tasks = [
    r
    for r in map(
        json.loads,
        (args.root / "data/prepared/sft_pool.jsonl").read_text().splitlines(),
    )
    if r["task_id"] not in public
]
tasks = [
    r
    for i, r in enumerate(tasks)
    if i % args.shards == args.shard and r["task_id"] not in completed
]
if args.round == 1:
    if not (out / f"generated-{args.shard}.complete").exists():
        raise RuntimeError(
            "Complete the first two candidates before the bounded second pair"
        )
    failed = {
        r["task_id"]
        for r in map(
            json.loads, (out / f"completed-{args.shard}.jsonl").read_text().splitlines()
        )
        if not r["accepted"]
    }
    tasks = [r for r in tasks if r["task_id"] in failed]
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
engine = Rollout(
    tokenizer, InferenceEndpoint(args.model), seed=4242 + args.shard + 1000 * args.round
)
t0 = time.monotonic()
n_accepted = 0
with accepted_path.open("a") as selected, traces.open("a") as raw, completed_path.open(
    "a"
) as done:
    for start in range(0, len(tasks), args.batch):
        batch = tasks[start : start + args.batch]
        states = engine.sample(batch, group_size=2)
        records = [engine.record(s) for s in states]
        for row in records:
            raw.write(json.dumps(row) + "\n")
        raw.flush()
        for i, task in enumerate(batch):
            chosen = None
            for row in records[2 * i : 2 * i + 2]:
                if not row["correct"] or row["invalid_calls"] or row["stop"] != "final":
                    continue
                if task["required_retrieval"] and not any(
                    e.get("tool") == "search" for e in row["events"]
                ):
                    continue
                try:
                    encoded = encode_sft(tokenizer, row["messages"], 6144)
                except ValueError:
                    continue
                chosen = dict(
                    task_id=task["task_id"],
                    family=task["family"],
                    messages=row["messages"],
                    **encoded,
                    provenance="same-base4B-bounded-four-candidate-rejection",
                    reliability="target_tools_reexecuted_final_verified",
                    candidate_id=row["candidate_id"],
                )
                break  # fixed first valid, not cheapest or best-of-test
            if chosen:
                selected.write(json.dumps(chosen) + "\n")
                n_accepted += 1
            done.write(
                json.dumps({"task_id": task["task_id"], "accepted": bool(chosen)})
                + "\n"
            )
        selected.flush()
        done.flush()
        print(
            json.dumps(
                {
                    "processed": min(start + len(batch), len(tasks)),
                    "total": len(tasks),
                    "accepted": n_accepted,
                    "elapsed_seconds": time.monotonic() - t0,
                    "physical_generated_tokens": sum(
                        r["generated_tokens"] for r in records
                    ),
                }
            ),
            flush=True,
        )
(out / f"generated-{suffix}.complete").write_text(
    json.dumps(
        {
            "round": args.round,
            "processed": len(tasks),
            "accepted_this_invocation": n_accepted,
            "seconds": time.monotonic() - t0,
        }
    )
)

engine.pool.shutdown()
