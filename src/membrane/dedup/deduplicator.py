"""Near-duplicate memory removal using embedding cosine similarity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable, Sequence

from membrane.firewall import count_tokens

logger = logging.getLogger(__name__)

_MODEL: Any = None
_MODEL_FAILED = False


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _hash_embed(text: str, dim: int = 32) -> list[float]:
    """Deterministic pseudo-embedding for environments without fastembed."""
    import hashlib

    h = hashlib.sha256(text.encode("utf-8")).digest()
    out = []
    for i in range(dim):
        out.append(((h[i % len(h)] + i) % 256) / 255.0 * 2.0 - 1.0)
    return out


def _get_model() -> Any:
    """Load the fastembed model once per process; ``None`` if unavailable."""
    global _MODEL, _MODEL_FAILED
    if _MODEL is None and not _MODEL_FAILED:
        try:
            from fastembed import TextEmbedding  # type: ignore

            _MODEL = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _MODEL_FAILED = True
            logger.warning(
                "fastembed unavailable; dedup falls back to hash embeddings", exc_info=True
            )
    return _MODEL


def _default_embed_batch(texts: list[str], dim: int = 32) -> list[list[float]]:
    model = _get_model()
    if model is not None:
        try:
            return [list(e) for e in model.embed(texts)]
        except Exception:
            logger.warning("fastembed embedding failed; using hash embeddings", exc_info=True)
    else:
        logger.warning("using hash embeddings for dedup (similarity is not semantic)")
    return [_hash_embed(t, dim) for t in texts]


@dataclass
class DeduplicationResult:
    kept: list[Any]
    removed: list[Any]
    clusters: list[list[int]]
    tokens_saved: int
    merge_map: dict[int, int]


class SemanticDeduplicator:
    def __init__(
        self,
        *,
        similarity_threshold: float = 0.85,
        merge_strategy: str = "keep_longest",
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
        embedding_cache: Any = None,
    ) -> None:
        self._threshold = similarity_threshold
        self._merge_strategy = merge_strategy
        self._embed_fn = embedding_fn or _default_embed_batch
        self._cache = embedding_cache

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._cache is not None:
            out: list[list[float]] = []
            for t in texts:
                out.append(
                    self._cache.get_or_compute(
                        t, lambda x: self._embed_fn([x])[0]
                    )
                )
            return out
        return self._embed_fn(texts)

    def deduplicate(
        self,
        memories: Sequence[Any],
        *,
        text_fn: Callable[[Any], str] = str,
        score_fn: Callable[[Any], float] | None = None,
    ) -> DeduplicationResult:
        texts = [text_fn(m) for m in memories]
        if not texts:
            return DeduplicationResult(
                kept=[], removed=[], clusters=[], tokens_saved=0, merge_map={}
            )
        vecs = self._embed(texts)
        n = len(memories)

        # compare each item against cluster representatives (first member) only, so
        # similarity is never chained transitively through intermediate items
        clusters_map: dict[int, list[int]] = {}
        for i in range(n):
            for rep, members in clusters_map.items():
                if _cosine(vecs[rep], vecs[i]) >= self._threshold:
                    members.append(i)
                    break
            else:
                clusters_map[i] = [i]

        kept_idx: list[int] = []
        removed_idx: list[int] = []
        merge_map: dict[int, int] = {}

        for _root, idxs in clusters_map.items():
            if len(idxs) == 1:
                kept_idx.append(idxs[0])
                continue
            scores = []
            for ii in idxs:
                m = memories[ii]
                sc = score_fn(m) if score_fn else 1.0
                scores.append((ii, sc, len(text_fn(m))))
            if self._merge_strategy == "keep_highest_score":
                best = max(scores, key=lambda x: x[1])[0]
            elif self._merge_strategy == "merge":
                best = min(idxs)
            else:
                best = max(scores, key=lambda x: x[2])[0]
            kept_idx.append(best)
            for ii in idxs:
                if ii != best:
                    removed_idx.append(ii)
                    merge_map[ii] = best

        kept = [memories[i] for i in sorted(kept_idx)]
        removed = [memories[i] for i in removed_idx]
        clusters = [sorted(v) for v in clusters_map.values() if len(v) > 1]
        tok_before = sum(count_tokens(text_fn(memories[i])) for i in range(n))
        tok_after = sum(count_tokens(text_fn(memories[i])) for i in sorted(kept_idx))
        saved = max(0, tok_before - tok_after)

        return DeduplicationResult(
            kept=kept,
            removed=removed,
            clusters=clusters,
            tokens_saved=saved,
            merge_map=merge_map,
        )
