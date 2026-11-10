"""Coverage tests for membrane.profile."""

import sys

import pytest

from membrane.profile import PRESET_NAMES, MembraneProfile


def test_minimal_is_empty():
    p = MembraneProfile.minimal()
    assert p.classifier is None and p.deduplicator is None and p.freshness is None
    assert p.policies is None
    assert p.enable_profiler is False
    assert p.trace_exporters == []
    assert p.include_ranking_diagnostics is True


def test_trace_exporters_not_shared():
    a, b = MembraneProfile(), MembraneProfile()
    a.trace_exporters.append(object())
    assert b.trace_exporters == []


def test_coding_preset():
    p = MembraneProfile.coding()
    assert p.classifier is not None
    assert p.deduplicator is not None
    assert p.freshness is not None
    assert p.policies is not None
    assert p.enable_profiler is True


def test_conversational_preset():
    p = MembraneProfile.conversational()
    assert p.classifier is not None and p.deduplicator is not None
    assert p.freshness is not None
    assert p.policies is None
    assert p.enable_profiler is True


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_preset_by_name(name):
    assert isinstance(MembraneProfile.preset(name), MembraneProfile)


def test_preset_normalises_and_defaults():
    assert MembraneProfile.preset("  CODING ").enable_profiler is True
    assert MembraneProfile.preset("").enable_profiler is False
    assert MembraneProfile.preset(None).enable_profiler is False  # type: ignore[arg-type]


def test_preset_unknown():
    with pytest.raises(ValueError, match="unknown profile 'nope'"):
        MembraneProfile.preset("nope")


def test_from_dict_sets_known_ignores_unknown_and_trace_exporters():
    p = MembraneProfile.from_dict(
        {
            "enable_lineage": True,
            "include_ranking_diagnostics": False,
            "bogus": 1,
            "trace_exporters": ["x"],
        }
    )
    assert p.enable_lineage is True
    assert p.include_ranking_diagnostics is False
    assert not hasattr(p, "bogus")
    assert p.trace_exporters == []


def test_override_returns_copy():
    base = MembraneProfile(enable_lineage=True)
    new = base.override(enable_profiler=True, reranker="r")
    assert new is not base
    assert new.enable_profiler is True and new.reranker == "r"
    assert new.enable_lineage is True
    assert base.enable_profiler is False and base.reranker is None


def test_override_list_is_copied():
    base = MembraneProfile()
    base.trace_exporters.append("e")
    new = base.override()
    new.trace_exporters.append("f")
    assert base.trace_exporters == ["e"]


def test_override_unknown_field():
    with pytest.raises(TypeError, match="no field 'zzz'"):
        MembraneProfile().override(zzz=1)


def test_override_preserves_subclass():
    class Sub(MembraneProfile):
        pass

    assert type(Sub().override(enable_lineage=True)) is Sub


def test_from_yaml_missing_pyyaml(monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ImportError, match="PyYAML"):
        MembraneProfile.from_yaml("whatever.yaml")


@pytest.mark.requires_extras
def test_from_yaml_file(tmp_path):
    pytest.importorskip("yaml")
    f = tmp_path / "p.yaml"
    f.write_text("enable_lineage: true\nunknown: 3\n", encoding="utf-8")
    p = MembraneProfile.from_yaml(str(f))
    assert p.enable_lineage is True
    empty = tmp_path / "e.yaml"
    empty.write_text("", encoding="utf-8")
    assert MembraneProfile.from_yaml(str(empty)).enable_lineage is False
    bad = tmp_path / "b.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        MembraneProfile.from_yaml(str(bad))
