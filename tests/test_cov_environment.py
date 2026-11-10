"""Coverage tests for membrane.environment."""

import sys
import types

import pytest

from membrane import environment as env


@pytest.fixture(autouse=True)
def _reset():
    env.clear_environment_snapshot()
    yield
    env.clear_environment_snapshot()


def test_default_none():
    assert env.get_current_environment_state() is None


def test_set_snapshot_fills_timestamp():
    env.set_environment_snapshot(latency_ms=12.5, tool_failure_count=2)
    snap = env.get_current_environment_state()
    assert snap.latency_ms == 12.5
    assert snap.tool_failure_count == 2
    assert snap.cpu_load is None
    assert snap.timestamp and "T" in snap.timestamp


def test_set_snapshot_explicit_timestamp_and_to_dict():
    env.set_environment_snapshot(cpu_load=0.3, loop_retries=1, timestamp="2020-01-01")
    assert env.get_current_environment_state().to_dict() == {
        "latency_ms": None,
        "cpu_load": 0.3,
        "tool_failure_count": None,
        "loop_retries": 1,
        "timestamp": "2020-01-01",
    }


def test_set_from_state_and_clear():
    s = env.EnvironmentSnapshot(latency_ms=1.0)
    env.set_environment_snapshot_from_state(s)
    assert env.get_current_environment_state() is s
    env.clear_environment_snapshot()
    assert env.get_current_environment_state() is None


def test_capture_cpu_load_with_fake_psutil(monkeypatch):
    fake = types.ModuleType("psutil")
    fake.cpu_percent = lambda interval=None: 25.0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert env.capture_cpu_load() == 0.25


def test_capture_cpu_load_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert env.capture_cpu_load() is None


def test_snapshot_to_stress_inputs():
    assert env.snapshot_to_stress_inputs(None) is None
    assert env.snapshot_to_stress_inputs(env.EnvironmentSnapshot(timestamp="t")) is None
    full = env.EnvironmentSnapshot(
        latency_ms=0.0, cpu_load=0.0, tool_failure_count=0, loop_retries=0, timestamp="t"
    )
    # zero values are kept (only None is dropped)
    assert env.snapshot_to_stress_inputs(full) == {
        "latency_ms": 0.0,
        "cpu_load": 0.0,
        "tool_failure_count": 0,
        "loop_retries": 0,
    }
    part = env.EnvironmentSnapshot(loop_retries=3)
    assert env.snapshot_to_stress_inputs(part) == {"loop_retries": 3}
