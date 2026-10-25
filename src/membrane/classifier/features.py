"""Text feature extraction for memory kind classification."""

from __future__ import annotations

import math
import re
from typing import List


# case-sensitive and code-shaped so prose like "I got this from John" does not match
_CODE_BLOCK = re.compile(
    r"```|`|\bdef \w+\(|\bclass \w+\s*[:(]|^\s*import [\w.]+(?:\s+as \w+)?\s*$"
    r"|^\s*from [\w.]+ import \w|\bSELECT\b[^;\n]*\bFROM\b",
    re.M,
)
_LIST = re.compile(r"^\s*[\d]+[\.\)]\s+|^\s*[-*]\s+", re.M)
_TEMPORAL = re.compile(
    r"\b(yesterday|last week|last month|remember when|i recall|conversation|you said)\b|\b(?:user|assistant):",
    re.I,
)
_DEF = re.compile(r"\b(is a |is an |defined as|refers to|means )\b", re.I)
_IMPERATIVE = re.compile(r"\b(always|never|must |should |do not|run |execute )\b", re.I)


def extract_features(text: str) -> List[float]:
    """Return a fixed-length feature vector (~15 features)."""
    t = text or ""
    lower = t.lower()
    n = max(len(t), 1)
    sentences = max(1, t.count(".") + t.count("!") + t.count("?"))
    words = max(1, len(t.split()))
    return [
        1.0 if _CODE_BLOCK.search(t) else 0.0,
        1.0 if _LIST.search(t) else 0.0,
        float(min(1.0, len(_TEMPORAL.findall(lower)) / 3.0)),
        float(min(1.0, len(_DEF.findall(lower)) / 2.0)),
        float(min(1.0, len(_IMPERATIVE.findall(lower)) / 3.0)),
        float(len(t) / n),
        float(words / sentences / 50.0),
        float(min(1.0, t.count("\n") / 20.0)),
        1.0 if "http://" in t or "https://" in t else 0.0,
        float(min(1.0, len(set(lower.split())) / max(words, 1))),
        1.0 if "?" in t else 0.0,
        float(t.count("#") / 10.0),
        1.0 if "step" in lower and ("1" in t or "2" in t) else 0.0,
        float(math.log1p(words) / 10.0),
        1.0 if "error" in lower or "traceback" in lower else 0.0,
    ]


FEATURE_DIM = 15
