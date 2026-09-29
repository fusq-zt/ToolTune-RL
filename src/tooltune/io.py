"""Small JSONL and model-identity helpers shared by command-line entries."""

import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def model_identity(path):
    """Bind calibration and resume state to the exact supplied base weights."""
    path = Path(path)
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError("Expected a local merged model with safetensors weights")
    digest = hashlib.sha256()
    for source in files + [path / "config.json"]:
        digest.update(source.name.encode())
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
