"""LRU embedding cache with optional Redis backend."""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal
from collections.abc import Callable


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


@dataclass
class _Entry:
    vector: list[float]
    expires_at: float | None


class EmbeddingCache:
    """
    Content-hash keyed cache for embedding vectors.

    ``backend='memory'`` uses a process-local LRU. ``backend='redis'`` stores
    JSON-serialized vectors under keys ``membrane:emb:{sha256}``. ``model_id`` namespaces keys
    per embedding model.
    """

    def __init__(
        self,
        *,
        max_size: int = 10_000,
        ttl_seconds: int | None = 3600,
        backend: Literal["memory", "redis"] = "memory",
        redis_client: Any | None = None,
        model_id: str | None = None,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._backend = backend
        self._redis = redis_client
        self._model_id = model_id
        self._lock = threading.Lock()
        self._lru: OrderedDict[str, _Entry] = OrderedDict()
        self._stats = CacheStats()

    def _use_redis(self) -> bool:
        return self._backend == "redis" and self._redis is not None

    def _key(self, text: str) -> str:
        # model_id namespaces keys so vectors from different embedders never collide
        if self._model_id:
            return _sha256_text(f"{self._model_id}\x00{text}")
        return _sha256_text(text)

    def _lookup(self, key: str) -> list[float] | None:
        """Return a cached vector or ``None``; does not touch hit/miss stats."""
        if self._use_redis():
            import json

            try:
                raw = self._redis.get(f"membrane:emb:{key}")  # type: ignore[union-attr]
                if raw:
                    return list(json.loads(raw))
            except Exception:
                pass
            return None
        with self._lock:
            ent = self._lru.get(key)
            if ent is None:
                return None
            if ent.expires_at is not None and time.time() > ent.expires_at:
                del self._lru[key]
                return None
            self._lru.move_to_end(key)
            return list(ent.vector)

    def _store(self, key: str, vec: list[float]) -> None:
        if self._use_redis():
            import json

            rkey = f"membrane:emb:{key}"
            try:
                payload = json.dumps(list(vec))
                if self._ttl:
                    self._redis.setex(rkey, self._ttl, payload)  # type: ignore[union-attr]
                else:
                    self._redis.set(rkey, payload)  # type: ignore[union-attr]
            except Exception:
                pass
            return
        exp = None if self._ttl is None else time.time() + float(self._ttl)
        with self._lock:
            self._lru[key] = _Entry(vector=list(vec), expires_at=exp)
            self._lru.move_to_end(key)
            while len(self._lru) > self._max_size:
                self._lru.popitem(last=False)
                self._stats.evictions += 1
            self._stats.size = len(self._lru)

    def _count(self, *, hits: int = 0, misses: int = 0) -> None:
        with self._lock:
            self._stats.hits += hits
            self._stats.misses += misses

    def get_or_compute(
        self,
        text: str,
        embed_fn: Callable[[str], list[float]],
    ) -> list[float]:
        key = self._key(text)
        cached = self._lookup(key)
        if cached is not None:
            self._count(hits=1)
            return cached
        self._count(misses=1)
        vec = embed_fn(text)  # computed without holding the lock
        self._store(key, vec)
        return list(vec)

    def get_or_compute_batch(
        self,
        texts: list[str],
        embed_fn: Callable[[list[str]], list[list[float]]],
    ) -> list[list[float]]:
        out: list[list[float]] = [[] for _ in texts]
        # identical texts in one batch are embedded once
        pending: OrderedDict[str, list[int]] = OrderedDict()
        pending_texts: list[str] = []
        hits = 0
        misses = 0
        for i, t in enumerate(texts):
            key = self._key(t)
            if key in pending:
                pending[key].append(i)
                misses += 1
                continue
            hit = self._lookup(key)
            if hit is not None:
                out[i] = hit
                hits += 1
            else:
                pending[key] = [i]
                pending_texts.append(t)
                misses += 1
        self._count(hits=hits, misses=misses)
        if pending_texts:
            computed = embed_fn(pending_texts)
            for (key, idxs), vec in zip(pending.items(), computed, strict=False):
                self._store(key, vec)
                for idx in idxs:
                    out[idx] = list(vec)
        return out

    def invalidate(self, text: str) -> None:
        key = self._key(text)
        if self._use_redis():
            with contextlib.suppress(Exception):
                self._redis.delete(f"membrane:emb:{key}")  # type: ignore[union-attr]
        with self._lock:
            self._lru.pop(key, None)
            self._stats.size = len(self._lru)

    def clear(self) -> None:
        with self._lock:
            self._lru.clear()
            self._stats.size = 0

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            self._stats.size = len(self._lru)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
            )
