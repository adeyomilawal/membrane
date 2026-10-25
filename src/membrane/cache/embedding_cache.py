"""LRU embedding cache with optional Redis backend."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional


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
    vector: List[float]
    expires_at: Optional[float]


class EmbeddingCache:
    """
    Content-hash keyed cache for embedding vectors.

    ``backend='memory'`` uses a process-local LRU. ``backend='redis'`` stores
    JSON-serialized vectors under keys ``membrane:emb:{sha256}``.
    """

    def __init__(
        self,
        *,
        max_size: int = 10_000,
        ttl_seconds: Optional[int] = 3600,
        backend: Literal["memory", "redis"] = "memory",
        redis_client: Optional[Any] = None,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._backend = backend
        self._redis = redis_client
        self._lock = threading.Lock()
        self._lru: "OrderedDict[str, _Entry]" = OrderedDict()
        self._stats = CacheStats()

    def _key(self, text: str) -> str:
        return _sha256_text(text)

    def get_or_compute(
        self,
        text: str,
        embed_fn: Callable[[str], List[float]],
    ) -> List[float]:
        key = self._key(text)
        if self._backend == "redis" and self._redis is not None:
            return self._redis_get_or_compute(key, text, embed_fn)
        with self._lock:
            now = time.time()
            if key in self._lru:
                ent = self._lru[key]
                if ent.expires_at is not None and now > ent.expires_at:
                    del self._lru[key]
                    self._stats.misses += 1
                else:
                    self._lru.move_to_end(key)
                    self._stats.hits += 1
                    return list(ent.vector)
            self._stats.misses += 1
            vec = embed_fn(text)
            exp = None if self._ttl is None else now + float(self._ttl)
            self._lru[key] = _Entry(vector=list(vec), expires_at=exp)
            self._lru.move_to_end(key)
            while len(self._lru) > self._max_size:
                self._lru.popitem(last=False)
                self._stats.evictions += 1
            self._stats.size = len(self._lru)
            return list(vec)

    def _redis_get_or_compute(
        self,
        key: str,
        text: str,
        embed_fn: Callable[[str], List[float]],
    ) -> List[float]:
        import json

        redis = self._redis
        if redis is None:
            return list(embed_fn(text))

        rkey = f"membrane:emb:{key}"
        try:
            raw = redis.get(rkey)
            if raw:
                self._stats.hits += 1
                return json.loads(raw)
        except Exception:
            pass
        self._stats.misses += 1
        vec = embed_fn(text)
        try:
            payload = json.dumps(vec)
            if self._ttl:
                redis.setex(rkey, self._ttl, payload)
            else:
                redis.set(rkey, payload)
        except Exception:
            pass
        return list(vec)

    def get_or_compute_batch(
        self,
        texts: List[str],
        embed_fn: Callable[[List[str]], List[List[float]]],
    ) -> List[List[float]]:
        out: List[List[float]] = []
        pending_idx: List[int] = []
        pending_texts: List[str] = []
        for i, t in enumerate(texts):
            key = self._key(t)
            hit: Optional[List[float]] = None
            if self._backend == "memory" or self._redis is None:
                with self._lock:
                    if key in self._lru:
                        ent = self._lru[key]
                        now = time.time()
                        if ent.expires_at is None or now <= ent.expires_at:
                            self._lru.move_to_end(key)
                            self._stats.hits += 1
                            hit = list(ent.vector)
            if hit is not None:
                out.append(hit)
            else:
                out.append([])  # placeholder
                pending_idx.append(i)
                pending_texts.append(t)
        if pending_texts:
            computed = embed_fn(pending_texts)
            for j, idx in enumerate(pending_idx):
                vec = computed[j]
                out[idx] = vec
                t = pending_texts[j]
                key = self._key(t)
                if self._backend == "memory" or self._redis is None:
                    now = time.time()
                    exp = None if self._ttl is None else now + float(self._ttl)
                    with self._lock:
                        self._lru[key] = _Entry(vector=list(vec), expires_at=exp)
                        self._lru.move_to_end(key)
                        while len(self._lru) > self._max_size:
                            self._lru.popitem(last=False)
                            self._stats.evictions += 1
                        self._stats.size = len(self._lru)
        return out

    def invalidate(self, text: str) -> None:
        key = self._key(text)
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
