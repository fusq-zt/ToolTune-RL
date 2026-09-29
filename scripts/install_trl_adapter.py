"""Enable external environment-token masks in the pinned TRL trainer.

Only masking interfaces are extended; GRPO advantages and loss remain upstream.
Source fragments below are from Apache-2.0 licensed Hugging Face TRL 0.27.2.
"""

import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path

MARKER = "# ToolTune: external rollout environment mask\n"
CHANGES = [
    (
        "            tool_mask = None\n",
        '            tool_mask = extra_fields.pop("env_mask", None)\n',
    ),
    (
        "        if self.tools:\n            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]",
        "        tool_mask = None\n        if tool_mask_list is not None:\n            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]",
    ),
    (
        "completion_mask if not self.tools else completion_mask * tool_mask",
        "completion_mask if tool_mask is None else completion_mask * tool_mask",
    ),
    (
        "completion_mask.bool() if not self.tools else (completion_mask * tool_mask).bool()",
        "completion_mask.bool() if tool_mask is None else (completion_mask * tool_mask).bool()",
    ),
    (
        '        if self.tools:\n            output["tool_mask"] = tool_mask',
        '        if tool_mask is not None:\n            output["tool_mask"] = tool_mask',
    ),
    (
        'completion_mask if not self.tools else completion_mask * inputs["tool_mask"]',
        'completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]',
    ),
]


def adapt_source(source):
    if source.startswith(MARKER):
        return source
    for before, after in CHANGES:
        if source.count(before) != 1:
            raise RuntimeError(
                "Expected an unmodified TRL 0.27.2 trainer; use a fresh environment"
            )
        source = source.replace(before, after)
    source = MARKER + source
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    import trl

    if trl.__version__ != "0.27.2":
        raise RuntimeError("Install trl==0.27.2 in the project virtual environment")
    path = (
        Path(importlib.util.find_spec("trl").origin).parent / "trainer/grpo_trainer.py"
    )
    source = path.read_text(encoding="utf-8")
    if args.check:
        if not source.startswith(MARKER):
            raise RuntimeError("Run this command without --check before training")
        print("TRL environment-mask adapter is present")
        return
    updated = adapt_source(source)
    if updated != source:
        backup = path.with_suffix(".py.tooltune-original")
        if backup.exists():
            raise RuntimeError(
                "Existing adapter backup: inspect the environment before changing it"
            )
        backup.write_text(source, encoding="utf-8")
        staging = path.with_suffix(".py.tooltune-tmp")
        staging.write_text(updated, encoding="utf-8")
        staging.replace(path)
    print("TRL adapter ready:", hashlib.sha256(updated.encode()).hexdigest())


if __name__ == "__main__":
    main()
