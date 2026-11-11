"""LOD engine: attention-shift detection via embedding centroid, and a fastembed embedder
for ``ContextCompiler(embedder=...)``."""

from __future__ import annotations

import logging
from typing import Any
from collections.abc import Callable, Sequence

from membrane import config as os_config

logger = logging.getLogger(__name__)

_embedding_model: Any = None
_numpy_warned = False


def _get_embedding_model():
    """Lazy-load fastembed (optional dependency)."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from fastembed import TextEmbedding

            _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        except Exception as e:
            logger.warning("fastembed not available for LOD (%s); attention_shift will be False", e)
            _embedding_model = False
    return _embedding_model


def fastembed_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> Callable[[Sequence[str]], list[list[float]]]:
    """An embedding function for ``ContextCompiler(embedder=...)``, backed by fastembed.

    Needs the ``lod`` extra (``pip install "membrane-aol[lod]"``). The model runs locally
    on the CPU; fastembed downloads it the first time it is used (about 130 MB for the
    default) and loads it when this function is called. Nothing is loaded or downloaded
    unless you call this.
    """
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ImportError(
            'fastembed is not installed; install it with: pip install "membrane-aol[lod]"'
        ) from exc
    model = TextEmbedding(model_name=model_name)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in model.embed([t.strip() or " " for t in texts])]

    return embed


def embed_texts(texts: Sequence[str]):
    """Embed strings; return (n, dim) numpy array or None on failure."""

    global _numpy_warned
    try:
        import numpy as np
    except ImportError:
        if not _numpy_warned:
            _numpy_warned = True
            logger.warning(
                "numpy not installed; attention-shift detection is off. "
                "Install membrane-aol[lod] to enable it."
            )
        return None

    model = _get_embedding_model()
    if model is False or not texts:
        return None
    try:
        vectors = list(model.embed([t.strip() or " " for t in texts]))
        return np.array([v.tolist() for v in vectors], dtype=np.float64)
    except Exception as e:
        logger.debug("LOD embed failed: %s", e)
        return None


def cosine_distance(a, b) -> float:
    """Cosine distance between two vectors (1 - cosine similarity)."""
    try:
        import numpy as np
    except ImportError:
        return 1.0

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size == 0:
        return 1.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def compute_attention_shift(
    last_user_messages: Sequence[str],
    current_user_message: str,
    threshold: float | None = None,
) -> bool:
    """
    True if current message is far from the centroid of the last user messages.
    """
    if threshold is None:
        threshold = os_config.get_lod_attention_shift_cosine_threshold()
    recent = list(last_user_messages)[-3:] if last_user_messages else []
    if not recent:
        return False
    texts = [*list(recent), current_user_message]
    emb = embed_texts(texts)
    if emb is None or emb.shape[0] < 2:
        return False
    try:
        import numpy as np
    except ImportError:
        return False

    centroid = np.mean(emb[:-1], axis=0)
    current_emb = emb[-1]
    dist = cosine_distance(current_emb, centroid)
    return dist > threshold


def compute_attention_shift_with_embeddings(
    last_user_message_embeddings: Sequence[Sequence[float]],
    current_user_embedding: Sequence[float],
    threshold: float | None = None,
) -> bool:
    """
    Attention shift using precomputed embeddings (caller supplies embedder).

    last_user_message_embeddings: up to last N user message vectors (same dim).
    current_user_embedding: vector for the current user message.
    """
    try:
        import numpy as np
    except ImportError:
        return False

    if threshold is None:
        threshold = os_config.get_lod_attention_shift_cosine_threshold()
    recent = list(last_user_message_embeddings)[-3:]
    if not recent:
        return False
    stack = np.array([*list(recent), list(current_user_embedding)], dtype=np.float64)
    if stack.shape[0] < 2:
        return False
    centroid = np.mean(stack[:-1], axis=0)
    current_emb = stack[-1]
    dist = cosine_distance(current_emb, centroid)
    return dist > threshold
