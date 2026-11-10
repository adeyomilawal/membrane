import pytest

from membrane import async_preflight as ap
from membrane.async_preflight import (
    get_preflight_cache,
    run_async_preflight,
    set_preflight_cache,
)


class FakeRedis:
    def __init__(self, fail=False):
        self.store = {}
        self.ex = None
        self.fail = fail

    async def get(self, key):
        if self.fail:
            raise RuntimeError("down")
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        if self.fail:
            raise RuntimeError("down")
        self.store[key] = value
        self.ex = ex


def test_key_helpers():
    assert ap.get_preflight_cache_key_prefix() == "membrane:preflight:"
    assert ap.get_preflight_cache_ttl_default() == 60
    assert ap._cache_key("c", "m") == ap._cache_key("c", "m")
    assert ap._cache_key("c", "m") != ap._cache_key("c", "n")
    assert ap._cache_key("a", "bc") != ap._cache_key("ab", "c")
    assert ap._full_key("c", "m").startswith("membrane:preflight:c:")


async def test_cache_roundtrip_and_ttl():
    r = FakeRedis()
    await set_preflight_cache(r, "c", "msg", "compressed", ttl_seconds=5)
    assert r.ex == 5
    assert await get_preflight_cache(r, "c", "msg") == "compressed"
    assert await get_preflight_cache(r, "c", "other") is None
    await set_preflight_cache(r, "c", "m2", "x")
    assert r.ex == 60


async def test_get_decodes_bytes_and_ignores_other_types():
    r = FakeRedis()
    key = ap._full_key("c", "m")
    r.store[key] = b"bytes!"
    assert await get_preflight_cache(r, "c", "m") == "bytes!"
    r.store[key] = 123
    assert await get_preflight_cache(r, "c", "m") is None


async def test_cache_noops_and_errors():
    r = FakeRedis()
    assert await get_preflight_cache(None, "c", "m") is None
    assert await get_preflight_cache(r, "", "m") is None
    await set_preflight_cache(r, "", "m", "x")
    await set_preflight_cache(r, "c", "m", "")
    await set_preflight_cache(None, "c", "m", "x")
    assert r.store == {}
    assert await get_preflight_cache(FakeRedis(fail=True), "c", "m") is None
    await set_preflight_cache(FakeRedis(fail=True), "c", "m", "x")


def make_fns(raw="word " * 200, compressed="short"):
    calls = {}

    async def retrieve(user_id, query):
        calls["retrieve"] = (user_id, query)
        return raw

    async def compress(text, tgt, msg):
        calls["compress"] = (text, tgt, msg)
        return compressed

    return retrieve, compress, calls


async def test_run_compresses_and_caches():
    r = FakeRedis()
    retrieve, compress, calls = make_fns()
    out = await run_async_preflight(
        "u", "c", "  hello  ", redis_client=r, retrieve_raw_middle=retrieve,
        compress_middle=compress, threshold=10, target_tokens=42,
    )
    assert out == "short"
    assert calls["retrieve"] == ("u", "hello")
    assert calls["compress"][1:] == (42, "  hello  ")
    assert await get_preflight_cache(r, "c", "  hello  ") == "short"


async def test_run_default_target_and_no_redis():
    retrieve, compress, calls = make_fns()
    out = await run_async_preflight(
        "u", "c", "hi", redis_client=None, retrieve_raw_middle=retrieve,
        compress_middle=compress, threshold=1,
    )
    assert out == "short"
    assert calls["compress"][1] == 500


async def test_run_below_threshold_returns_none():
    retrieve, compress, calls = make_fns(raw="tiny")
    out = await run_async_preflight(
        "u", "c", "hi", redis_client=FakeRedis(), retrieve_raw_middle=retrieve,
        compress_middle=compress,
    )
    assert out is None and "compress" not in calls


@pytest.mark.parametrize(
    ("user", "conv", "msg"),
    [("u", "c", ""), ("u", "c", "   "), ("", "c", "m"), ("u", "", "m")],
)
async def test_run_invalid_inputs(user, conv, msg):
    retrieve, compress, calls = make_fns()
    out = await run_async_preflight(
        user, conv, msg, redis_client=None, retrieve_raw_middle=retrieve,
        compress_middle=compress,
    )
    assert out is None and not calls


async def test_run_empty_raw_and_failures():
    retrieve, compress, _ = make_fns(raw="  ")
    assert await run_async_preflight(
        "u", "c", "m", redis_client=None, retrieve_raw_middle=retrieve,
        compress_middle=compress, threshold=0) is None

    async def boom(*a):
        raise RuntimeError("x")

    retrieve, _, _ = make_fns()
    assert await run_async_preflight(
        "u", "c", "m", redis_client=None, retrieve_raw_middle=retrieve,
        compress_middle=boom, threshold=0) is None


async def test_run_empty_compressed_not_cached():
    r = FakeRedis()
    retrieve, compress, _ = make_fns(compressed="")
    out = await run_async_preflight(
        "u", "c", "m", redis_client=r, retrieve_raw_middle=retrieve,
        compress_middle=compress, threshold=0)
    assert out == "" and r.store == {}
