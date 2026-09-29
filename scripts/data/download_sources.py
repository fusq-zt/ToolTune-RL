"""Fetch pinned public source assets into a user-selected workspace."""

import argparse
import json
from pathlib import Path
from tooltune.io import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--sources", type=Path, default=Path("configs/sources.json"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    sources = load_config(args.sources)
    metadata = args.root / "sources/metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    for item in sources["datasets"]:
        repo = item["repo"]
        snapshot_download(
            repo,
            repo_type="dataset",
            revision=item["revision"],
            allow_patterns=item["patterns"],
            local_dir=args.root / "data/raw" / repo.replace("/", "--"),
        )
        (metadata / (repo.replace("/", "--") + ".json")).write_text(
            json.dumps({"id": repo, "sha": item["revision"]}), encoding="utf-8"
        )
    if args.model_output:
        snapshot_download(
            sources["model"]["repo"],
            revision=sources["model"]["revision"],
            local_dir=args.model_output,
        )


if __name__ == "__main__":
    main()
