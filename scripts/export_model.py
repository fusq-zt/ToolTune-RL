"""Merge one adapter, save, reload and measure numerical consistency."""

import argparse
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--adapter", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
args = parser.parse_args()
if args.output.exists():
    raise RuntimeError("Choose a new immutable export directory")
tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    args.base,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    local_files_only=True,
).to(args.device)
model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
ids = tokenizer(
    "Compare the record values, calculate their sum, and explain your answer.",
    return_tensors="pt",
).to(args.device)
with torch.no_grad():
    before = model(**ids).logits[:, -4:].float().cpu()
merged = model.merge_and_unload(safe_merge=True)
merged.save_pretrained(args.output, safe_serialization=True)
tokenizer.save_pretrained(args.output)
del model, merged
torch.cuda.empty_cache()
reloaded = (
    AutoModelForCausalLM.from_pretrained(
        args.output,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    .to(args.device)
    .eval()
)
with torch.no_grad():
    after = reloaded(**ids).logits[:, -4:].float().cpu()
diff = (before - after).abs()
report = {
    "base": args.base,
    "adapter": args.adapter,
    "max_abs_logit_difference": float(diff.max()),
    "mean_abs_logit_difference": float(diff.mean()),
    "greedy_last4_equal": bool(torch.equal(before.argmax(-1), after.argmax(-1))),
}
(args.output / "merge-check.json").write_text(json.dumps(report, indent=2))
if (
    not torch.isfinite(after).all()
    or diff.mean() > 0.05
    or not report["greedy_last4_equal"]
):
    raise RuntimeError("Merge numerical consistency requires inspection")
print(json.dumps(report), flush=True)
