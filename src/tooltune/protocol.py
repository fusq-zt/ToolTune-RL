"""Qwen native function calls; token provenance is kept at append time."""

import json
import re

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Run isolated Python for computation. Print the result. No network or files are available.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search only the fixed documents available for this question.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]
SYSTEM = (
    "Solve the question accurately. You may use the provided tools when helpful. "
    "Search only accesses this question's local documents. All record values must come from those documents. "
    "Put only your final answer, without explanation, in <final_answer>...</final_answer>. "
    "Any reasoning must precede that final answer. Do not put a tool call in the final answer."
)
EFFICIENT = " Prefer the fewest tool calls that preserve correctness. Do not guess missing document values."


def messages_for(task, efficient=False):
    # Explicit whitelist. Never interpolate task repr, reference, split, or metadata.
    return [
        {"role": "system", "content": SYSTEM + (EFFICIENT if efficient else "")},
        {"role": "user", "content": str(task["question"])},
    ]


def prompt_ids(tokenizer, task, efficient=False):
    return tokenizer.apply_chat_template(
        messages_for(task, efficient),
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=True,
    )


def parse_action(text):
    calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    finals = re.findall(r"<final_answer>\s*(.*?)\s*</final_answer>", text, re.S)
    if calls:
        if len(calls) != 1 or finals:
            return "invalid", {
                "error": "Exactly one tool call or one final answer is required."
            }
        try:
            call = json.loads(calls[0])
            name, args = call["name"], call["arguments"]
            key = {"python": "code", "search": "query"}[name]
            if (
                not isinstance(args, dict)
                or set(args) != {key}
                or not isinstance(args[key], str)
            ):
                raise ValueError("invalid arguments")
            return name, args
        except (ValueError, KeyError, TypeError):
            return "invalid", {"error": "Invalid function name or JSON arguments."}
    if len(finals) == 1 and finals[0].strip() and "<tool_call>" not in text:
        return "final", finals[0].strip()
    return "invalid", {"error": "Missing or ambiguous final answer/tool call."}


def observation_ids(tokenizer, result):
    # Qwen template's exact standalone tool block after an assistant im_end.
    # Template compatibility is asserted against apply_chat_template in GPU preflight.
    text = (
        "\n<|im_start|>user\n<tool_response>\n"
        + result
        + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    )
    return tokenizer.encode(text, add_special_tokens=False)


def encode_sft(tokenizer, messages, maximum):
    """Derive masks from roles and template boundaries, not regex over token text."""
    ids = tokenizer.apply_chat_template(
        messages[:2], tools=TOOLS, tokenize=True, add_generation_prompt=True
    )
    mask = [0] * len(ids)
    history = messages[:2]
    for message in messages[2:]:
        if message["role"] == "assistant":
            extended = tokenizer.apply_chat_template(
                history + [message],
                tools=TOOLS,
                tokenize=True,
                add_generation_prompt=False,
            )
            if extended[: len(ids)] != ids:
                raise ValueError(
                    "Chat-template prefix changed; do not decode/re-encode generated IDs"
                )
            mask.extend([1] * (len(extended) - len(ids)))
            # The template appends a newline after EOS; inference never sampled it.
            newline = tokenizer.encode("\n", add_special_tokens=False)
            if extended[-len(newline) :] != newline:
                raise ValueError("Unexpected assistant delimiter")
            mask[-len(newline) :] = [0] * len(newline)
            ids = extended
        elif message["role"] == "tool":
            extended = tokenizer.apply_chat_template(
                history + [message],
                tools=TOOLS,
                tokenize=True,
                add_generation_prompt=True,
            )
            if extended[: len(ids)] != ids:
                raise ValueError("Tool template prefix mismatch")
            mask.extend([0] * (len(extended) - len(ids)))
            ids = extended
        else:
            raise ValueError("Unexpected trajectory role")
        history.append(message)
    if len(ids) > maximum:
        raise ValueError(
            "overlength: retain audit entry; never truncate away the answer"
        )
    if not any(mask):
        raise ValueError("No model tokens")
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [tok if m else -100 for tok, m in zip(ids, mask)],
    }
