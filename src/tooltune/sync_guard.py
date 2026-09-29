"""Preserve frozen BF16 parameters across official TRL's merge/unmerge sync.

The official sync remains responsible for communication and cache invalidation.
An immutable CPU copy prevents repeated floating-point add/subtract roundoff from
changing the disabled-adapter reference. All experimental groups use this guard.
"""

import json
from pathlib import Path
import time
import torch


def install_sync_guard(trainer, tokenizer, output):
    assert hasattr(trainer.model, "disable_adapter")
    assert not trainer.is_fsdp_enabled and trainer.accelerator.num_processes == 1
    frozen = {
        name: p.detach().cpu().clone()
        for name, p in trainer.model.named_parameters()
        if not p.requires_grad
    }
    count = sum(t.numel() * t.element_size() for t in frozen.values())
    original = trainer._move_model_to_vllm
    ids = tokenizer(
        "Read the records carefully and return their sum.", return_tensors="pt"
    ).input_ids.to(trainer.model.device)

    def reference_logits():
        was_training = trainer.model.training
        trainer.model.eval()
        with torch.no_grad(), trainer.model.disable_adapter():
            result = (
                trainer.model(input_ids=ids, use_cache=False, logits_to_keep=2)
                .logits.detach()
                .cpu()
            )
        trainer.model.train(was_training)
        return result

    reference = reference_logits()
    sync_count = 0

    def guarded_sync():
        nonlocal sync_count
        started = time.monotonic()
        pre_restore_equal = None
        restored_equal = None
        try:
            result = original()
            if sync_count < 3:
                pre_restore_equal = torch.equal(reference_logits(), reference)
        finally:
            # If communication failed while merged, clear adapter merge bookkeeping too.
            if any(getattr(m, "merged", False) for m in trainer.model.modules()):
                trainer.model.unmerge_adapter()
            with torch.no_grad():
                for name, p in trainer.model.named_parameters():
                    if name in frozen:
                        p.copy_(frozen[name])
        if sync_count < 3:
            restored_equal = torch.equal(reference_logits(), reference)
            assert (
                restored_equal
            ), "Disabled-adapter reference changed despite restoration"
        sync_count += 1
        with (Path(output) / "sync-reference-audit.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "sync": sync_count,
                        "step": trainer.state.global_step,
                        "cpu_snapshot_bytes": count,
                        "pre_restore_reference_exact": pre_restore_equal,
                        "restored_reference_exact": restored_equal,
                        "sync_and_guard_seconds": time.monotonic() - started,
                    }
                )
                + "\n"
            )
        return result

    trainer._move_model_to_vllm = guarded_sync
    return {
        "frozen_parameters": len(frozen),
        "cpu_snapshot_bytes": count,
        "reference_probe_positions": 2,
    }
