"""Contracts for reviewed, matched demonstrations used in efficiency SFT.

The review fields are caller attestations, not an automatic correctness judge.
This module never selects demonstrations using evaluation results.
"""

import math

from .protocol import encode_sft


def validate_pairs(rows, paired_rows, matched_rows=181, replay_rows=85):
    """Require the same task order and shared replay; identical pairs are valid."""
    expected = matched_rows + replay_rows
    if (
        min(matched_rows, replay_rows) < 1
        or len(rows) != expected
        or len(paired_rows) != expected
    ):
        raise ValueError("Expected the configured matched and replay row counts")
    for data in (rows, paired_rows):
        ids = [row.get("task_id") for row in data]
        if (
            any(not isinstance(key, str) or not key for key in ids)
            or len(set(ids)) != expected
        ):
            raise ValueError("Task IDs must be nonempty and unique within each arm")
        for i, row in enumerate(data):
            kind = "matched" if i < matched_rows else "replay"
            if row.get("kind") != kind:
                raise ValueError("Place matched demonstrations before shared replay")
            if (
                row.get("training_split") != "train"
                or row.get("reviewed_correct") is not True
            ):
                raise ValueError(
                    "Only previously reviewed correct training demonstrations are accepted"
                )
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError("Expected a complete system/user/assistant trajectory")
            roles = [message.get("role") for message in messages]
            if roles[:2] != ["system", "user"] or roles[2::2] != ["assistant"] * len(
                roles[2::2]
            ):
                raise ValueError(
                    "Expected an assistant action after each question or observation"
                )
            if roles[3::2] != ["tool"] * len(roles[3::2]) or roles[-1] != "assistant":
                raise ValueError(
                    "Tools must be followed by an assistant action ending in a final answer"
                )
    for row, other in zip(rows, paired_rows):
        if (row["task_id"], row["kind"], row["messages"][:2]) != (
            other["task_id"],
            other["kind"],
            other["messages"][:2],
        ):
            raise ValueError(
                "The two arms must use the same ordered questions and ordinary prompts"
            )
        if row["kind"] == "replay" and row["messages"] != other["messages"]:
            raise ValueError("Shared tool replay must be identical across arms")


def encode_reviewed_rows(tokenizer, rows, max_length=6144):
    """Rebuild assistant-only labels from the real template; never truncate."""
    encoded = []
    for row in rows:
        rebuilt = encode_sft(tokenizer, row["messages"], max_length)
        for key, values in rebuilt.items():
            if key in row and row[key] != values:
                raise ValueError(
                    f"{row['task_id']}: supplied {key} differs from role-based reconstruction"
                )
        if rebuilt["labels"][0] != -100:
            raise ValueError(
                "The initial prompt token must be excluded from causal loss"
            )
        encoded.append({"task_id": row["task_id"], **rebuilt})
    return encoded


class AccumulationCheck:
    """Check actual Trainer token denominators, including an incomplete tail group."""

    def __init__(self, rows, epochs=2, accumulation=16):
        if min(rows, epochs, accumulation) < 1:
            raise ValueError(
                "Positive dataset size, epochs and accumulation are required"
            )
        self.rows = rows
        self.accumulation = accumulation
        self.updates_per_epoch = math.ceil(rows / accumulation)
        self.total_updates = epochs * self.updates_per_epoch
        self.step = 0
        self.pending = []

    def add(self, supervised_tokens, reported_denominator, actual_microbatches):
        if supervised_tokens <= 0 or reported_denominator is None:
            raise RuntimeError(
                "Require official sum CE normalized by accumulated supervised tokens"
            )
        self.pending.append(
            (supervised_tokens, reported_denominator, actual_microbatches)
        )

    def finish(self, step):
        if step != self.step + 1 or step > self.total_updates:
            raise RuntimeError("Unexpected optimizer step")
        tail = self.rows % self.accumulation or self.accumulation
        size = tail if step % self.updates_per_epoch == 0 else self.accumulation
        denominator = sum(item[0] for item in self.pending)
        if len(self.pending) != size or any(
            den != denominator or batches != size for _, den, batches in self.pending
        ):
            raise RuntimeError(
                "Accumulation size or actual supervised-token denominator is incorrect"
            )
        self.pending.clear()
        self.step = step
        return {"step": step, "examples": size, "supervised_tokens": denominator}
