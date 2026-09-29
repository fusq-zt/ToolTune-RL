"""Correctness-gated bounded cost. Groups must be complete and contiguous."""

import math


def efficiency_weight(step, total_steps, maximum=0.10):
    progress = step / max(total_steps, 1)
    return maximum * min(max((progress - 0.2) / 0.8, 0.0), 1.0)


def group_rewards(
    rows, *, group_size=4, max_calls=5, mu=0.02, lam=0.10, efficiency=True
):
    if len(rows) % group_size:
        raise ValueError("Incomplete rollout group")
    output = []
    seen = set()
    for start in range(0, len(rows), group_size):
        group = rows[start : start + group_size]
        group_id = group[0]["group_id"]
        if any(r["group_id"] != group_id for r in group) or group_id in seen:
            raise ValueError("Noncontiguous or duplicate group")
        seen.add(group_id)
        if len({r["candidate_id"] for r in group}) != group_size:
            raise ValueError("Duplicate candidate ID")
        best = min((r["logical_calls"] for r in group if r["correct"]), default=None)
        for row in group:
            if row.get("infrastructure_error"):
                raise RuntimeError("Do not train on infrastructure failures")
            a = int(row["correct"])
            violation = min(max(row["invalid_calls"] / max_calls, 0), 1)
            reward = a - mu * violation
            if efficiency and a and best is not None:
                reward -= lam * min(
                    max((row["logical_calls"] - best) / max_calls, 0), 1
                )
            if not math.isfinite(reward):
                raise ValueError("Nonfinite reward")
            output.append(reward)
    return output
