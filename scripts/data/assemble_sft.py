"""Merge accepted demonstrations, validate masks, and write a deterministic SFT file."""

import argparse
import hashlib
import json
from pathlib import Path
from tooltune.io import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [row for path in args.inputs for row in read_jsonl(path)]
    if not rows or len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty, unique SFT question IDs")
    for row in rows:
        ids, attention, labels = (
            row[k] for k in ("input_ids", "attention_mask", "labels")
        )
        if not len(ids) == len(attention) == len(labels) or len(ids) > 6144:
            raise ValueError("Invalid SFT sequence length")
        if not any(label != -100 for label in labels):
            raise ValueError("No supervised model tokens")
        if any(label not in (-100, token) for token, label in zip(ids, labels)):
            raise ValueError("Label/token alignment mismatch")
    rows.sort(key=lambda r: hashlib.sha256(("42:" + r["task_id"]).encode()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Accepted SFT examples: {len(rows)}")


if __name__ == "__main__":
    main()
