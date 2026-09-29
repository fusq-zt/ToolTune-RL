"""Strict whole-final answers; no substring numeric or broad exception fallback."""

import collections
from fractions import Fraction
import re
import string


def normalized_qa(value):
    # Standard SQuAD/Hotpot normalization removes punctuation (does not add spaces).
    value = "".join(c for c in str(value).lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def qa_scores(prediction, reference):
    p, r = normalized_qa(prediction), normalized_qa(reference)
    em = bool(p) and p == r
    if p in {"yes", "no", "noanswer"} or r in {"yes", "no", "noanswer"}:
        return int(em), float(em)
    pt, rt = p.split(), r.split()
    overlap = sum((collections.Counter(pt) & collections.Counter(rt)).values())
    f1 = 2 * overlap / (len(pt) + len(rt)) if pt and rt else 0.0
    return int(em), f1


def numeric(value):
    s = str(value).strip().replace("−", "-")
    if s.startswith("\\boxed{") and s.endswith("}"):
        s = s[7:-1].strip()
    # Commas must be genuine thousands separators, not a list of answers.
    if "," in s:
        if not re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", s):
            return None
        s = s.replace(",", "")
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[-+]?\d+)?", s):
        return None
    try:
        return Fraction(s.replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return None


def verify(prediction, reference, kind):
    if kind == "qa":
        return bool(qa_scores(prediction, reference)[0])
    if kind == "numeric":
        p, r = numeric(prediction), numeric(reference)
        return p is not None and r is not None and p == r
    if kind == "math":
        from math_verify import parse, verify as math_verify
        from math_verify.parser import LatexExtractionConfig

        # Only parse a single bounded final expression. Library's timeout is retained.
        if not prediction.strip() or len(prediction) > 1024:
            return False

        def expression(text):
            return parse(
                "\\boxed{" + str(text) + "}",
                extraction_config=[LatexExtractionConfig()],
                extraction_mode="first_match",
            )

        return bool(
            math_verify(
                expression(reference), expression(prediction), timeout_seconds=3
            )
        )
    raise ValueError("Unknown verifier kind: " + kind)
