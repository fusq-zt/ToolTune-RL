"""CPU-only illustration of retrieval and relative rewards; no model generation."""

import json
from pathlib import Path
from tooltune.protocol import parse_action
from tooltune.rewards import group_rewards
from tooltune.tools.search import LocalDocument, LocalSearch

task = json.loads(Path(__file__).with_name("task.json").read_text(encoding="utf-8"))
action = '<tool_call>{"name":"search","arguments":{"query":"Luma Vex"}}</tool_call>'
name, arguments = parse_action(action)
print("Illustrative tool action:", name, arguments)
search = LocalSearch([LocalDocument(**doc) for doc in task["documents"]])
print(search.search(arguments["query"]))

# Authored examples explain the reward calculation, not measured model outputs.
candidates = [
    dict(
        group_id="example",
        candidate_id=0,
        correct=True,
        logical_calls=1,
        invalid_calls=0,
    ),
    dict(
        group_id="example",
        candidate_id=1,
        correct=True,
        logical_calls=3,
        invalid_calls=0,
    ),
    dict(
        group_id="example",
        candidate_id=2,
        correct=False,
        logical_calls=0,
        invalid_calls=0,
    ),
    dict(
        group_id="example",
        candidate_id=3,
        correct=False,
        logical_calls=1,
        invalid_calls=1,
    ),
]
print("\nIllustrative candidate rewards:")
for row, reward in zip(candidates, group_rewards(candidates)):
    print(
        f"  candidate={row['candidate_id']} correct={row['correct']} calls={row['logical_calls']} reward={reward:.3f}"
    )
