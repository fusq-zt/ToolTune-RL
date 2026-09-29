"""Adapter to the same vLLM sampling API on each independent trainer GPU."""

import json
import math
from pathlib import Path
import time
import torch
from vllm import SamplingParams


class ColocatedEndpoint:
    def __init__(self, trainer, output):
        self.trainer = trainer
        self.output = Path(output)
        self.retries = []
        self.sleep_events = []
        self.first_audit = False

    def begin(self):
        started = time.monotonic()
        # Official GRPO has already woken weights and synchronized the policy.
        torch.cuda.empty_cache()
        self.trainer.llm.wake_up(tags=["kv_cache"])
        torch.cuda.synchronize()
        self.sleep_events.append(
            dict(
                event="wake_kv_cache",
                step=self.trainer.state.global_step,
                seconds=time.monotonic() - started,
            )
        )

    def generate(self, prompts, seeds, maximum, temperature):
        assert len(prompts) == len(seeds)
        params = [
            SamplingParams(
                n=1,
                temperature=temperature,
                top_p=1.0,
                top_k=-1,
                max_tokens=maximum,
                seed=int(seed),
                logprobs=0,
                skip_special_tokens=False,
            )
            for seed in seeds
        ]
        outputs = self.trainer.llm.generate(
            [{"prompt_token_ids": p} for p in prompts],
            sampling_params=params,
            use_tqdm=False,
        )
        result = dict(prompt_ids=[], completion_ids=[], logprobs=[])
        for prompt, request in zip(prompts, outputs, strict=True):
            assert request.prompt_token_ids == prompt and len(request.outputs) == 1
            completion = request.outputs[0]
            ids = list(completion.token_ids)
            lps = [
                float(values[token].logprob)
                for token, values in zip(ids, completion.logprobs, strict=True)
            ]
            assert ids and all(math.isfinite(v) for v in lps)
            result["prompt_ids"].append(prompt)
            result["completion_ids"].append(ids)
            result["logprobs"].append(lps)
        return result

    def sleep(self):
        torch.cuda.synchronize()
        before = torch.cuda.mem_get_info()[0]
        started = time.monotonic()
        self.trainer.llm.sleep(level=2)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after = torch.cuda.mem_get_info()[0]
        self.sleep_events.append(
            dict(
                event="sleep_level2",
                step=self.trainer.state.global_step,
                seconds=time.monotonic() - started,
                freed_device_bytes=after - before,
                free_device_bytes=after,
            )
        )
        with (self.output / "colocation-memory.jsonl").open("a") as f:
            for row in self.sleep_events:
                f.write(json.dumps(row) + "\n")
        self.sleep_events.clear()

    def finish(self, states):
        self.sleep()
        if not self.first_audit:
            differences = []
            model = self.trainer.model
            was_training = model.training
            model.eval()
            with torch.no_grad():
                for state in states[:4]:
                    positions = [i for i, m in enumerate(state.mask) if m][:32]
                    if not positions:
                        continue
                    tokens = state.prompt + state.ids[: max(positions) + 1]
                    indices = torch.tensor(
                        [len(state.prompt) + i - 1 for i in positions],
                        device=model.device,
                    )
                    logits = (
                        model(
                            input_ids=torch.tensor([tokens], device=model.device),
                            use_cache=False,
                            logits_to_keep=indices,
                        )
                        .logits[0]
                        .float()
                        / 0.8
                    )
                    ids = torch.tensor(
                        [state.ids[i] for i in positions], device=model.device
                    )
                    actual = (
                        logits.log_softmax(-1)
                        .gather(-1, ids[:, None])
                        .squeeze(-1)
                        .cpu()
                        .tolist()
                    )
                    differences.extend(
                        abs(x - state.logprobs[i])
                        for x, i in zip(actual, positions, strict=True)
                    )
            model.train(was_training)
            assert differences
            report = dict(
                n=len(differences),
                mean_abs_difference=sum(differences) / len(differences),
                max_abs_difference=max(differences),
                threshold_mean=0.05,
            )
            report["passed"] = report["mean_abs_difference"] < report["threshold_mean"]
            (self.output / "colocation-logprob-check.json").write_text(
                json.dumps(report, indent=2)
            )
            assert report["passed"], "Single-card sampler/policy logprobs disagree"
            self.first_audit = True

    def sync_for_evaluation(self):
        # Match the official pre-rollout weight lifecycle after sleep(level=2).
        torch.cuda.empty_cache()
        self.trainer.llm.wake_up(tags=["weights"])
        self.trainer.llm.collective_rpc("reload_weights")
        self.trainer._move_model_to_vllm()
        self.begin()


class InferenceEndpoint:
    """The same token-ID / sampled-logprob interface for data generation and eval."""

    def __init__(self, model, memory_fraction=0.80):
        from vllm import LLM

        self.llm = LLM(
            model=model,
            dtype="bfloat16",
            max_model_len=6144,
            gpu_memory_utilization=memory_fraction,
            logprobs_mode="processed_logprobs",
        )
        self.retries = []

    def generate(self, prompts, seeds, maximum, temperature):
        params = [
            SamplingParams(
                n=1,
                temperature=temperature,
                top_p=1.0,
                top_k=-1,
                max_tokens=maximum,
                seed=int(seed),
                logprobs=0,
                skip_special_tokens=False,
            )
            for seed in seeds
        ]
        outputs = self.llm.generate(
            [{"prompt_token_ids": p} for p in prompts],
            sampling_params=params,
            use_tqdm=False,
        )
        result = dict(prompt_ids=[], completion_ids=[], logprobs=[])
        for prompt, request in zip(prompts, outputs, strict=True):
            assert request.prompt_token_ids == prompt and len(request.outputs) == 1
            completion = request.outputs[0]
            ids = list(completion.token_ids)
            logprobs = [
                float(values[token].logprob)
                for token, values in zip(ids, completion.logprobs, strict=True)
            ]
            assert ids and all(math.isfinite(value) for value in logprobs)
            result["prompt_ids"].append(prompt)
            result["completion_ids"].append(ids)
            result["logprobs"].append(logprobs)
        return result
