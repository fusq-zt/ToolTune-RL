"""Align Tool-Star to SFT-only questions, adapt native calls, replay Python."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from transformers import AutoTokenizer
from tooltune.data.toolstar import validate_transcript
from tooltune.protocol import messages_for, encode_sft
from tooltune.sandbox import preflight, execute
from tooltune.verifier import verify


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()
preflight()
prepared = args.root / "data/prepared"
tasks = {
    norm(t["question"]): t
    for t in map(json.loads, (prepared / "sft_pool.jsonl").read_text().splitlines())
}
others = {
    norm(t["question"])
    for name in ["rl", "dev_config", "dev_checkpoint", "test"]
    for t in map(json.loads, (prepared / (name + ".jsonl")).read_text().splitlines())
}
source = (
    args.root / "data/raw/dongguanting--Tool-Star-SFT-54K/final_sft_edition9_v2.json"
)
rows = json.loads(source.read_text())
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
out = args.root / "data/sft"
out.mkdir(exist_ok=True)
stats = Counter()
selected = []
seen = set()
for row in rows:
    question = str(row.get("input", "")).strip()
    key = norm(question)
    if key in others:
        stats["evaluation_or_rl_overlap_quarantined"] += 1
        continue
    if key not in tasks:
        stats["unmatched_or_not_sft_pool_quarantined"] += 1
        continue
    task = tasks[key]
    if task["task_id"] in seen:
        stats["duplicate_question"] += 1
        continue
    text = str(row.get("output", ""))
    validation = validate_transcript(text, max_tool_calls=5)
    if not validation.valid or "<search>" in text:
        stats["invalid_or_unreplayable_search"] += 1
        continue
    try:
        correct = verify(validation.final_answer, task["answer"], task["verifier"])
    except Exception as exc:
        stats["verifier_error"] += 1
        continue
    if not correct:
        stats["incorrect_final"] += 1
        continue
    messages = messages_for(task)
    reasoning = ""
    pending = False
    replayed = 0
    valid = True
    for match in re.finditer(
        r"<(think|python|result|answer)>\s*(.*?)\s*</\1>", text, re.S
    ):
        kind, body = match.groups()
        if kind == "think":
            reasoning += body.replace("\\n", "\n").strip() + "\n"
        elif kind == "python":
            # Source corpus uses escaped line breaks in the serialized tool program.
            code = body.replace("\\n", "\n").strip()
            result = execute(code)
            if not result["ok"]:
                valid = False
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": reasoning.strip(),
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "python", "arguments": {"code": code}},
                        }
                    ],
                }
            )
            reasoning = ""
            pending = True
            replayed += 1
            actual = result["output"].strip()
        elif kind == "result":
            if not pending or " ".join(body.replace("\\n", "\n").split()) != " ".join(
                actual.split()
            ):
                valid = False
                break
            messages.append({"role": "tool", "name": "python", "content": actual})
            pending = False
        elif kind == "answer":
            messages.append(
                {
                    "role": "assistant",
                    "content": reasoning
                    + "<final_answer>"
                    + validation.final_answer
                    + "</final_answer>",
                }
            )
    if not valid or pending:
        stats["execution_or_observation_mismatch"] += 1
        continue
    try:
        encoded = encode_sft(tokenizer, messages, 6144)
    except ValueError:
        stats["length_or_template_rejection"] += 1
        continue
    selected.append(
        dict(
            task_id=task["task_id"],
            family=task["family"],
            messages=messages,
            **encoded,
            provenance="Tool-Star-SFT-54K@f85b4fe0809a30a3b360b6f61689e462f79e2ec1",
            source_content_sha256=hashlib.sha256(
                json.dumps(row, sort_keys=True).encode()
            ).hexdigest(),
            reliability=(
                "python_reexecuted_final_verified"
                if replayed
                else "source_format_final_verified_no_tool"
            ),
            replayed_calls=replayed
        )
    )
    seen.add(task["task_id"])
    stats["accepted"] += 1
    if len(selected) % 100 == 0:
        print(json.dumps(dict(stats)), flush=True)
(out / "public.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
(out / "public-audit.json").write_text(
    json.dumps(
        {
            "counts": dict(stats),
            "family": dict(Counter(r["family"] for r in selected)),
            "alignment": "exact normalized original question in SFT-only pool; unmatched excluded",
            "limitation": "final-answer validation does not prove all reasoning is correct",
        },
        indent=2,
    )
)
print(json.dumps(dict(stats)), flush=True)
