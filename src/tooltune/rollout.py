"""Batched Qwen tool trajectories and bounded local branching; TRL owns training."""

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import time
from tooltune.tools.search import LocalDocument, LocalSearch
from . import sandbox
from .protocol import prompt_ids, observation_ids, parse_action, messages_for
from .verifier import verify


def stable_seed(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") % (
        2**31 - 1
    )


@dataclass
class PathState:
    task: dict
    group_id: str
    candidate_id: int
    root_id: int
    prompt: list
    ids: list = field(default_factory=list)
    mask: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    events: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    logical_calls: int = 0
    physical_calls: int = 0
    invalid_calls: int = 0
    generated_tokens: int = 0
    prefill_tokens: int = 0
    branch_point: int | None = None
    final_answer: str = ""
    stop: str | None = None
    elapsed: float = 0.0


class Rollout:
    def __init__(
        self,
        tokenizer,
        endpoint,
        *,
        seed=42,
        temperature=0.8,
        max_calls=5,
        total_tokens=6144,
        model_tokens=2048,
        step_tokens=768,
    ):
        self.tokenizer = tokenizer
        self.endpoint = endpoint
        self.seed = seed
        self.temperature = temperature
        self.max_calls = max_calls
        self.total_tokens = total_tokens
        self.model_tokens = model_tokens
        self.step_tokens = step_tokens
        self.history = defaultdict(lambda: deque(maxlen=2048))
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.batch_number = 0
        self.probe_stats = []

    def new(self, task, group, candidate, efficient=False):
        return PathState(
            task,
            group,
            candidate,
            candidate,
            prompt_ids(self.tokenizer, task, efficient),
            messages=messages_for(task, efficient),
        )

    def tool(self, state, name, args):
        started = time.monotonic()
        if name == "python":
            result = sandbox.execute(args["code"])
        else:
            search = LocalSearch(
                [LocalDocument(**d) for d in state.task.get("documents", [])]
            )
            output = search.search(args["query"])
            result = {
                "ok": not output.startswith("SEARCH_ERROR"),
                "output": output,
                "status": "ok",
            }
        return dict(result, seconds=time.monotonic() - started)

    def complete(self, states):
        started = time.monotonic()
        for turn in range(self.max_calls + 1):
            active = []
            for s in states:
                if s.stop:
                    continue
                remaining = min(
                    self.model_tokens - sum(s.mask),
                    self.total_tokens - len(s.prompt) - len(s.ids),
                )
                if remaining <= 0:
                    s.stop = "token_budget"
                else:
                    active.append((s, min(self.step_tokens, remaining)))
            if not active:
                break
            buckets = defaultdict(list)
            for state, budget in active:
                # Never round up beyond any member's hard budget.
                buckets[budget].append(state)
            pending = []
            for budget, cohort in buckets.items():
                contexts = [s.prompt + s.ids for s in cohort]
                seeds = [
                    stable_seed(
                        f"{self.seed}:{self.batch_number}:{s.group_id}:{s.candidate_id}:{len(s.ids)}"
                    )
                    for s in cohort
                ]
                result = self.endpoint.generate(
                    contexts, seeds, budget, self.temperature
                )
                for s, ids, lps in zip(
                    cohort, result["completion_ids"], result["logprobs"], strict=True
                ):
                    s.prefill_tokens += len(s.prompt) + len(s.ids)
                    start = len(s.ids)
                    s.ids.extend(ids)
                    s.mask.extend([1] * len(ids))
                    s.logprobs.extend(lps)
                    s.generated_tokens += len(ids)
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                    action, args = parse_action(text)
                    s.messages.append({"role": "assistant", "content": text})
                    if action == "final":
                        s.final_answer = args
                        s.stop = "final"
                        s.elapsed = time.monotonic() - started
                    elif action == "invalid":
                        s.invalid_calls += 1
                        s.stop = "protocol_error"
                    elif s.logical_calls >= self.max_calls:
                        s.invalid_calls += 1
                        s.stop = "call_budget"
                    elif ids[-1] != self.tokenizer.eos_token_id:
                        s.stop = "turn_truncation"
                    else:
                        s.logical_calls += 1
                        pending.append(
                            (
                                s,
                                action,
                                args,
                                start,
                                self.pool.submit(self.tool, s, action, args),
                            )
                        )
            for s, action, args, start, future in pending:
                result = future.result()  # infrastructure errors must propagate
                s.physical_calls += 1
                s.invalid_calls += int(not result["ok"])
                env = observation_ids(self.tokenizer, result["output"])
                if len(s.prompt) + len(s.ids) + len(env) >= self.total_tokens:
                    s.stop = "observation_budget"
                    s.events.append(
                        dict(
                            tool=action, args=args, **result, discarded_observation=True
                        )
                    )
                    continue
                s.ids.extend(env)
                s.mask.extend([0] * len(env))
                s.logprobs.extend([0.0] * len(env))
                s.events.append(
                    dict(
                        tool=action,
                        args=args,
                        **result,
                        token_offset=len(s.ids),
                        model_start=start,
                    )
                )
                s.messages.append(
                    {"role": "tool", "name": action, "content": result["output"]}
                )
                s.observations.append(
                    dict(
                        offset=len(s.ids),
                        tool=action,
                        event_count=len(s.events),
                        message_count=len(s.messages),
                        calls=s.logical_calls,
                        invalid=s.invalid_calls,
                        model_start=start,
                    )
                )
            for s in states:
                if s.stop and s.elapsed == 0.0:
                    s.elapsed = time.monotonic() - started
        for s in states:
            if not s.stop:
                s.stop = "turn_budget"
            if s.elapsed == 0.0:
                s.elapsed = time.monotonic() - started
            assert len(s.ids) == len(s.mask) == len(s.logprobs)
        return states

    def entropy(self, model, state, observation):
        """Full-vocabulary entropy of sampling distribution; FP32 reduction."""
        import torch

        probe_started = time.monotonic()
        start = observation["offset"]
        after = [
            i for i in range(start, min(start + 16, len(state.ids))) if state.mask[i]
        ]
        before = [
            i
            for i in range(
                observation["model_start"], min(observation["model_start"] + 16, start)
            )
            if state.mask[i]
        ]
        if not after or not before:
            return None
        positions = sorted(set(before + after))
        absolute = [len(state.prompt) + i - 1 for i in positions]
        input_ids = torch.tensor(
            [state.prompt + state.ids[: max(positions) + 1]], device=model.device
        )
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits = (
                model(
                    input_ids=input_ids,
                    use_cache=False,
                    logits_to_keep=torch.tensor(absolute, device=model.device),
                )
                .logits[0]
                .float()
                / self.temperature
            )
            probs = logits.softmax(-1)
            hs = logits.logsumexp(-1) - (probs * logits).sum(-1)
            assert torch.isfinite(hs).all()
            values = dict(zip(positions, hs.cpu().tolist()))
        model.train(was_training)
        self.probe_stats.append(
            {
                "input_tokens": int(input_ids.numel()),
                "positions": len(positions),
                "seconds": time.monotonic() - probe_started,
                "tool": observation["tool"],
            }
        )
        return sum(values[i] for i in after) / len(after) - sum(
            values[i] for i in before
        ) / len(before)

    def sample(
        self,
        tasks,
        *,
        group_size=4,
        mode="full",
        model=None,
        efficient=False,
        branch_budget=None,
    ):
        if mode not in {"full", "random", "entropy", "calibrated"}:
            raise ValueError(mode)
        if branch_budget is not None and not 0 <= branch_budget <= group_size // 2:
            raise ValueError("Branch budget must be between zero and half the group")
        if branch_budget == 0:
            mode = "full"
        self.batch_number += 1
        roots = (
            group_size
            if mode == "full"
            else group_size
            - (group_size // 2 if branch_budget is None else branch_budget)
        )
        groups = [
            [
                self.new(t, f'{self.batch_number}:{t["task_id"]}', i, efficient)
                for i in range(roots)
            ]
            for t in tasks
        ]
        self.complete([s for group in groups for s in group])
        pending = []
        rng = random.Random(stable_seed(f"{self.seed}:{self.batch_number}:branch"))
        calibration_updates = []
        for task, group in zip(tasks, groups):
            candidates = []
            for root in group:
                for obs in root.observations:
                    if obs["offset"] >= len(root.ids):
                        continue
                    delta = (
                        self.entropy(model, root, obs)
                        if mode in {"entropy", "calibrated"}
                        else None
                    )
                    if mode in {"entropy", "calibrated"} and delta is None:
                        continue
                    if mode == "random":
                        score = rng.random()
                    elif mode in {"entropy", "calibrated"}:
                        hist = (
                            self.history[obs["tool"]]
                            if mode == "calibrated"
                            and len(self.history[obs["tool"]]) >= 32
                            else self.history["global"]
                        )
                        score = (
                            sum(x <= delta for x in hist) / len(hist) if hist else 0.5
                        )
                        calibration_updates.append((obs["tool"], delta))
                    else:
                        continue
                    obs.update(
                        delta_entropy=delta, trigger_score=score, triggered=score >= 0.5
                    )
                    # Predeclared median trigger. Independent roots fill unused slots.
                    if score >= 0.5:
                        candidates.append((score, rng.random(), root, obs, delta))
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            for i in range(roots, group_size):
                if candidates:
                    score, _, root, obs, delta = candidates.pop(0)
                    child = copy.deepcopy(root)
                    child.candidate_id = i
                    child.root_id = root.root_id
                    child.branch_point = obs["offset"]
                    child.ids = child.ids[: obs["offset"]]
                    child.mask = child.mask[: obs["offset"]]
                    child.logprobs = child.logprobs[: obs["offset"]]
                    child.events = child.events[: obs["event_count"]]
                    child.messages = child.messages[: obs["message_count"]]
                    child.observations = []
                    child.logical_calls = obs["calls"]
                    child.invalid_calls = obs["invalid"]
                    child.physical_calls = child.generated_tokens = (
                        child.prefill_tokens
                    ) = 0
                    child.final_answer = ""
                    child.stop = None
                    child.elapsed = 0.0
                    child.events = copy.deepcopy(child.events)
                    child.events.append(
                        {
                            "branch_delta_entropy": delta,
                            "branch_score": score,
                            "shared_prefix": True,
                        }
                    )
                else:
                    child = self.new(task, group[0].group_id, i, efficient)
                group.append(child)
                pending.append(child)
        self.complete(pending)
        for tool, delta in calibration_updates:
            self.history["global"].append(delta)
            self.history[tool].append(delta)
        return [s for group in groups for s in group]

    def record(self, s):
        return dict(
            task_id=s.task["task_id"],
            family=s.task["family"],
            cluster_id=s.task["cluster_id"],
            group_id=s.group_id,
            candidate_id=s.candidate_id,
            root_id=s.root_id,
            branch_point=s.branch_point,
            correct=(
                verify(s.final_answer, s.task["answer"], s.task["verifier"])
                if s.final_answer
                else False
            ),
            final_answer=s.final_answer,
            stop=s.stop,
            logical_calls=s.logical_calls,
            physical_calls=s.physical_calls,
            invalid_calls=s.invalid_calls,
            generated_tokens=s.generated_tokens,
            prefill_tokens=s.prefill_tokens,
            required_retrieval=s.task.get("required_retrieval", False),
            prompt_ids=s.prompt,
            completion_ids=s.ids,
            env_mask=s.mask,
            logprobs=s.logprobs,
            events=s.events,
            observations=s.observations,
            messages=s.messages,
            elapsed=s.elapsed,
            latency_scope="batched_cohort_arrival_to_turn_completion; branch excludes prefix construction",
            repeated_calls=len([e for e in s.events if "tool" in e])
            - len(
                {
                    json.dumps([e["tool"], e["args"]], sort_keys=True)
                    for e in s.events
                    if "tool" in e
                }
            ),
        )
