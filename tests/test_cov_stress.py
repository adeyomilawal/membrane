import pytest

from membrane.stress import (
    compute_stress_from_environment,
    get_recent_stress_for_user,
    set_recent_stress_for_user,
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


@pytest.mark.parametrize("env", [None, {}, [], "x", {"unknown": 1}, {"latency_ms": None}])
def test_compute_none_for_unusable(env):
    assert compute_stress_from_environment(env) is None


def test_compute_components_and_clamping():
    assert compute_stress_from_environment({"latency_ms": 250}) == 0.5
    assert compute_stress_from_environment({"latency_ms": 5000}) == 1.0
    assert compute_stress_from_environment({"cpu_load": -3}) == 0.0
    assert compute_stress_from_environment({"cpu_load": 7}) == 1.0
    assert compute_stress_from_environment({"tool_failure_count": 1}) == pytest.approx(0.2)
    assert compute_stress_from_environment({"loop_retries": 10}) == 1.0


def test_compute_averages_components():
    env = {"latency_ms": 500, "cpu_load": 0.0, "tool_failure_count": 5, "loop_retries": 0}
    assert compute_stress_from_environment(env) == pytest.approx(0.5)


async def test_set_then_get_roundtrip(monkeypatch):
    monkeypatch.setenv("MEMBRANE_STRESS_KEY_PREFIX", "t:")
    r = FakeRedis()
    await set_recent_stress_for_user(r, "u", 0.4)
    assert r.store == {"t:u": "0.4"}
    assert r.ex == 300
    assert await get_recent_stress_for_user(r, "u") == 0.4


async def test_set_clamps(monkeypatch):
    r = FakeRedis()
    await set_recent_stress_for_user(r, "u", 9)
    assert list(r.store.values()) == ["1.0"]
    await set_recent_stress_for_user(r, "v", -1)
    assert await get_recent_stress_for_user(r, "v") == 0.0


async def test_noop_without_client_or_user():
    r = FakeRedis()
    await set_recent_stress_for_user(None, "u", 0.5)
    await set_recent_stress_for_user(r, "", 0.5)
    assert r.store == {}
    assert await get_recent_stress_for_user(None, "u") is None
    assert await get_recent_stress_for_user(r, "") is None


async def test_get_missing_and_errors():
    r = FakeRedis()
    assert await get_recent_stress_for_user(r, "nobody") is None
    r.store["membrane:stress:u"] = "garbage"
    assert await get_recent_stress_for_user(r, "u") is None
    r.store["membrane:stress:u"] = "5"
    assert await get_recent_stress_for_user(r, "u") == 1.0
    assert await get_recent_stress_for_user(FakeRedis(fail=True), "u") is None


async def test_set_swallows_errors():
    await set_recent_stress_for_user(FakeRedis(fail=True), "u", 0.5)
