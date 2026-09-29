"""Move saved autograd tensors to host RAM for long contexts; retain official loss."""

import json
from pathlib import Path
import torch


def install_long_context_offload(trainer, output, threshold=4096):
    original = trainer._compute_loss
    trainer._colocation_original_loss = original
    trainer._colocation_offload_threshold = threshold

    def wrapped(model, inputs):
        length = inputs["prompt_ids"].shape[1] + inputs["completion_ids"].shape[1]
        if length >= trainer._colocation_offload_threshold:
            # PyTorch's saved-tensor hook preserves tensor values and the autograd graph.
            # Backward transparently copies saved tensors back to their original device.
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                loss = original(model, inputs)
            with (Path(output) / "long-context-offload.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            step=trainer.state.global_step,
                            total_tokens=length,
                            threshold=threshold,
                            mode="saved_tensors_cpu_pinned",
                        )
                    )
                    + "\n"
                )
            return loss
        return original(model, inputs)

    trainer._compute_loss = wrapped
    return dict(
        threshold_tokens=threshold,
        mode="torch.autograd.graph.save_on_cpu",
        pin_memory=True,
        official_loss_unchanged=True,
        token_limits_unchanged=True,
    )
