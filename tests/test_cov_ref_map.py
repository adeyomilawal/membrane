"""Coverage tests for membrane.ref_map."""

import json
import sys
import types

import pytest

from membrane import ref_map as rm


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.set_calls = []
        self.closed = False

    async def set(self, key, val, ex=None):
        self.set_calls.append((key, val, ex))
        self.store[key] = val

    async def get(self, key):
        return self.store.get(key)

    async def aclose(self):
        self.closed = True


class BoomRedis:
    async def set(self, *a, **k):
        raise RuntimeError("boom")

    async def get(self, *a, **k):
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        "MEMBRANE_REF_MAP_PREFIX",
        "MEMORY_OS_KERNEL_REF_MAP_PREFIX",
        "MEMBRANE_REF_MAP_TTL_SECONDS",
        "MEMORY_OS_KERNEL_REF_MAP_TTL_SECONDS",
        "MEMBRANE_REDIS_URL",
        "MEMORY_OS_KERNEL_REDIS_URL",
        "REDIS_URL",
    ):
        monkeypatch.delenv(k, raising=False)


def test_serialize_roundtrip_and_none_type():
    s = rm._serialize_ref_maps({1: "a", 2: "b"}, None)
    assert json.loads(s) == {"ref_map": {"1": "a", "2": "b"}, "ref_map_type": {}}
    assert rm._deserialize_ref_maps(s) == ({1: "a", 2: "b"}, {})
    s2 = rm._serialize_ref_maps({1: "a"}, {1: "fact"})
    assert rm._deserialize_ref_maps(s2) == ({1: "a"}, {1: "fact"})


@pytest.mark.parametrize(
    "data",
    ["not json", json.dumps({"ref_map": {"x": "a"}}), "{bad"],
)
def test_deserialize_invalid_returns_empty(data):
    assert rm._deserialize_ref_maps(data) == ({}, {})


def test_deserialize_null_sections():
    assert rm._deserialize_ref_maps('{"ref_map": null}') == ({}, {})


def test_key_with_and_without_user(monkeypatch):
    assert rm._ref_map_key("c1", None) == "membrane:ref_map:c1"
    assert rm._ref_map_key("c1", "u9") == "membrane:ref_map:u9:c1"
    monkeypatch.setenv("MEMBRANE_REF_MAP_PREFIX", "p:")
    assert rm._ref_map_key("c1", "u9") == "p:u9:c1"


async def test_set_get_roundtrip_with_client():
    r = FakeRedis()
    await rm.set_ref_map_for_conversation("c", {1: "g1"}, {1: "t"}, redis_client=r, user_id="u")
    key, _, ttl = r.set_calls[0]
    assert key == "membrane:ref_map:u:c"
    assert ttl == 600
    got = await rm.get_ref_map_for_conversation("c", user_id="u", redis_client=r)
    assert got == ({1: "g1"}, {1: "t"})
    # different user does not see it
    assert await rm.get_ref_map_for_conversation("c", user_id="v", redis_client=r) == ({}, {})
    # a caller-supplied client is not closed
    assert r.closed is False


async def test_set_explicit_and_env_ttl(monkeypatch):
    r = FakeRedis()
    await rm.set_ref_map_for_conversation("c", {}, ttl_seconds=5, redis_client=r)
    assert r.set_calls[-1][2] == 5
    monkeypatch.setenv("MEMBRANE_REF_MAP_TTL_SECONDS", "42")
    await rm.set_ref_map_for_conversation("c", {}, redis_client=r)
    assert r.set_calls[-1][2] == 42
    # empty map is stored, replacing stale one
    assert r.store["membrane:ref_map:c"]


async def test_set_ignores_bad_input():
    r = FakeRedis()
    await rm.set_ref_map_for_conversation("", {1: "a"}, redis_client=r)
    await rm.set_ref_map_for_conversation("c", [1], redis_client=r)  # type: ignore[arg-type]
    assert r.set_calls == []


async def test_get_empty_conversation_id_and_miss():
    r = FakeRedis()
    assert await rm.get_ref_map_for_conversation("", redis_client=r) == ({}, {})
    assert await rm.get_ref_map_for_conversation("nope", redis_client=r) == ({}, {})


async def test_redis_errors_are_swallowed():
    b = BoomRedis()
    await rm.set_ref_map_for_conversation("c", {1: "a"}, redis_client=b)
    assert await rm.get_ref_map_for_conversation("c", redis_client=b) == ({}, {})


async def test_no_client_no_url_skips():
    await rm.set_ref_map_for_conversation("c", {1: "a"})
    assert await rm.get_ref_map_for_conversation("c") == ({}, {})


async def test_redis_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "redis.asyncio", None)
    monkeypatch.setitem(sys.modules, "redis", None)
    await rm.set_ref_map_for_conversation("c", {1: "a"}, redis_url="redis://x")
    assert await rm.get_ref_map_for_conversation("c", redis_url="redis://x") == ({}, {})


async def test_from_url_client_created_and_closed(monkeypatch):
    fake = FakeRedis()
    seen = {}

    def from_url(url, decode_responses):
        seen["url"] = url
        seen["decode"] = decode_responses
        return fake

    mod = types.ModuleType("redis.asyncio")
    mod.from_url = from_url  # type: ignore[attr-defined]
    pkg = types.ModuleType("redis")
    pkg.asyncio = mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "redis", pkg)
    monkeypatch.setitem(sys.modules, "redis.asyncio", mod)

    await rm.set_ref_map_for_conversation("c", {1: "a"}, redis_url="redis://explicit")
    assert seen == {"url": "redis://explicit", "decode": True}
    assert fake.closed is True

    fake.closed = False
    monkeypatch.setenv("REDIS_URL", "redis://fromenv")
    got = await rm.get_ref_map_for_conversation("c")
    assert seen["url"] == "redis://fromenv"
    assert got == ({1: "a"}, {})
    assert fake.closed is True


async def test_get_swallows_non_object_payload():
    r = FakeRedis()
    r.store["membrane:ref_map:c"] = "[1, 2]"
    assert await rm.get_ref_map_for_conversation("c", redis_client=r) == ({}, {})


def test_deserialize_rejects_non_object_payloads():
    assert rm._deserialize_ref_maps("[1, 2]") == ({}, {})
    assert rm._deserialize_ref_maps("null") == ({}, {})
