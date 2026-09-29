"""Checkpoint-only evaluation, isolated from training sampler state."""

from collections import defaultdict
import json
from pathlib import Path
import time
from tooltune.rollout import Rollout


def evaluate_checkpoint(tokenizer, endpoint, tasks, output, batch=16):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    engine = Rollout(tokenizer, endpoint, temperature=0.0)
    started = time.monotonic()
    rows = []
    with (output / "predictions.jsonl").open("w") as f:
        for start in range(0, len(tasks), batch):
            for state in engine.sample(tasks[start : start + batch], group_size=1):
                row = engine.record(state)
                rows.append(row)
                f.write(json.dumps(row) + "\n")
            f.flush()
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    report = {
        "n": len(rows),
        "macro_accuracy": sum(
            sum(r["correct"] for r in rs) / len(rs) for rs in families.values()
        )
        / len(families),
        "micro_accuracy": sum(r["correct"] for r in rows) / len(rows),
        "families": {
            k: sum(r["correct"] for r in rs) / len(rs) for k, rs in families.items()
        },
        "logical_calls": sum(r["logical_calls"] for r in rows),
        "invalid_calls": sum(r["invalid_calls"] for r in rows),
        "generated_tokens": sum(r["generated_tokens"] for r in rows),
        "prefill_tokens": sum(r["prefill_tokens"] for r in rows),
        "physical_calls": sum(r["physical_calls"] for r in rows),
        "seconds": time.monotonic() - started,
        "protocol_failures": sum(r["stop"] == "protocol_error" for r in rows),
        "retry_records": engine.endpoint.retries,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    engine.pool.shutdown()
    return report


def is_better(candidate, incumbent):
    """Preregistered 0.1 pp accuracy tie, then fewer calls, then earlier step."""
    if incumbent is None:
        return True
    delta = candidate["macro_accuracy"] - incumbent["macro_accuracy"]
    if abs(delta) > 0.001:
        return delta > 0
    return (candidate["logical_calls"], candidate["step"]) < (
        incumbent["logical_calls"],
        incumbent["step"],
    )
