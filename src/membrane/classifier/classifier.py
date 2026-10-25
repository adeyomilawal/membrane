"""Logistic regression memory classifier (no external model download)."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections.abc import Sequence

from membrane.classifier.features import FEATURE_DIM, extract_features

logger = logging.getLogger(__name__)


class MemoryKind(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    INSTRUCTIONAL = "instructional"
    CONVERSATIONAL = "conversational"
    UNKNOWN = "unknown"


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _default_weights() -> list[list[float]]:
    """Hand-tuned weights: rows = classes (episodic, semantic, procedural, instructional,
    conversational).
    """
    rng = FEATURE_DIM
    rows = []
    # episodic: temporal + conversational markers
    w_ep = [2.5, 0.0, 3.0, -0.5, -1.0] + [0.0] * (rng - 5)
    # semantic: definitions
    w_se = [-1.0, 0.0, -1.0, 3.5, -0.5] + [0.2] * (rng - 5)
    # procedural: code + lists + steps
    w_pr = [3.0, 2.5, -1.0, -1.0, 0.5] + [0.15] * (rng - 5)
    # instructional: imperatives
    w_in = [-0.5, 0.5, -0.5, 0.0, 3.0] + [0.1] * (rng - 5)
    # conversational
    w_co = [1.0, 0.5, 2.0, -0.5, -0.5] + [0.2] * (rng - 5)
    rows = [w_ep, w_se, w_pr, w_in, w_co]
    for r in rows:
        while len(r) < rng:
            r.append(0.0)
        del r[rng:]
    return rows


def _load_weights_path() -> Path:
    return Path(__file__).resolve().parent / "weights.json"


@dataclass
class ClassificationResult:
    kind: MemoryKind
    confidence: float
    features: dict[str, float]


class MemoryClassifier:
    """Multiclass one-vs-rest logistic regression on extracted features."""

    _KIND_ORDER = (
        MemoryKind.EPISODIC,
        MemoryKind.SEMANTIC,
        MemoryKind.PROCEDURAL,
        MemoryKind.INSTRUCTIONAL,
        MemoryKind.CONVERSATIONAL,
    )

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.35,
        fallback_kind: MemoryKind = MemoryKind.SEMANTIC,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.fallback_kind = fallback_kind
        self._weights: list[list[float]] = _default_weights()
        self._biases: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
        p = _load_weights_path()
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                w = data.get("weights")
                b = data.get("biases")
                if isinstance(w, list) and len(w) == 5:
                    self._weights = [list(map(float, row)) for row in w]
                if isinstance(b, list) and len(b) == 5:
                    self._biases = [float(x) for x in b]
            except Exception:
                logger.warning(
                    "failed to load classifier weights from %s; using defaults", p, exc_info=True
                )

    def classify(self, text: str) -> ClassificationResult:
        feats = extract_features(text)
        scores: list[float] = []
        for i in range(5):
            z = self._biases[i] + sum(
                self._weights[i][j] * feats[j] for j in range(min(len(feats), FEATURE_DIM))
            )
            scores.append(_sigmoid(z))
        best_i = max(range(5), key=lambda i: scores[i])
        conf = scores[best_i]
        kind = self._KIND_ORDER[best_i]
        if conf < self.confidence_threshold:
            kind = self.fallback_kind
            conf = max(scores)
        feat_dict = {f"f{i}": feats[i] for i in range(len(feats))}
        return ClassificationResult(kind=kind, confidence=float(conf), features=feat_dict)

    def classify_batch(self, texts: Sequence[str]) -> list[ClassificationResult]:
        return [self.classify(t) for t in texts]
