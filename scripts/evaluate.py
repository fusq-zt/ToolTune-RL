"""Evaluate a local merged model with the common greedy single-trajectory protocol."""

import argparse
from pathlib import Path
from tooltune.io import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from tooltune.endpoints import InferenceEndpoint
    from tooltune.evaluation import evaluate_checkpoint
    from tooltune.sandbox import preflight

    if args.output.exists():
        raise FileExistsError("Choose a new evaluation directory")
    tasks = read_jsonl(args.data)
    if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Expected nonempty, unique evaluation task IDs")
    preflight()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    report = evaluate_checkpoint(
        tokenizer, InferenceEndpoint(args.model), tasks, args.output
    )
    print(
        f"n={report['n']}; macro={report['macro_accuracy']:.4%}; calls={report['logical_calls']}"
    )


if __name__ == "__main__":
    main()
