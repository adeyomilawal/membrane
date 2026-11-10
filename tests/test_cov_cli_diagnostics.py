"""Coverage tests for membrane.cli.diagnostics."""

import json
from types import SimpleNamespace

import pytest

from membrane.cli import diagnostics as d
from membrane.profile import MembraneProfile


def test_parse_memory_arg_plain():
    m = d.parse_memory_arg("just text", 4)
    assert (m.id, m.content, m.priority, m.memory_type) == ("mem-4", "just text", 2, "semantic")


@pytest.mark.parametrize(
    "tier,mtype", [("1", "procedural"), ("2", "semantic"), ("3", "episodic")]
)
def test_parse_memory_arg_tiers(tier, mtype):
    m = d.parse_memory_arg(f"id1|{tier}|body|with|pipes", 1)
    assert m.id == "id1"
    assert m.content == "body|with|pipes"
    assert m.priority == int(tier)
    assert m.memory_type == mtype


def test_parse_memory_arg_empty_id_uses_index():
    assert d.parse_memory_arg("|1|x", 7).id == "mem-7"


def test_parse_memory_arg_bad_tier():
    with pytest.raises(ValueError, match="must be an integer"):
        d.parse_memory_arg("a|x|c", 1)
    with pytest.raises(ValueError, match="must be 1, 2, or 3"):
        d.parse_memory_arg("a|4|c", 1)


def test_load_memories_args_only_and_none():
    assert d.load_memories(None, None) == []
    ms = d.load_memories(["a|1|x", "plain"], None)
    assert [m.id for m in ms] == ["a", "mem-2"]


def test_load_memories_json(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(
        json.dumps(
            [
                "s1|3|from string",
                {"id": "o1", "content": "obj", "priority": 1},
                {"payload": {"b": 1, "a": 2}, "semantic_tier": 3},
                {"content": "typed", "memory_type": "episodic"},
            ]
        )
    )
    ms = d.load_memories(["cli"], str(p))
    assert [m.id for m in ms] == ["mem-1", "s1", "o1", "mem-4", "mem-5"]
    by_id = {m.id: m for m in ms}
    assert by_id["s1"].memory_type == "episodic"
    assert by_id["o1"].memory_type == "procedural"
    assert by_id["mem-4"].content == '{"a": 2, "b": 1}'  # non-str payload serialised sorted
    assert by_id["mem-4"].priority == 3
    assert by_id["mem-5"].memory_type == "episodic"
    assert by_id["mem-5"].priority == 2


def test_load_memories_json_errors(tmp_path):
    p = tmp_path / "m.json"
    p.write_text('{"not": "list"}')
    with pytest.raises(ValueError, match="must contain a list"):
        d.load_memories(None, str(p))
    p.write_text("[1]")
    with pytest.raises(ValueError, match="strings or objects"):
        d.load_memories(None, str(p))
    p.write_text('[{"content": "x", "priority": 9}]')
    with pytest.raises(ValueError, match="1, 2, or 3"):
        d.load_memories(None, str(p))
    p.write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        d.load_memories(None, str(p))


def test_resolve_profile_name():
    assert d.resolve_profile_name(None) is None
    assert d.resolve_profile_name("") is None
    assert isinstance(d.resolve_profile_name("minimal"), MembraneProfile)
    with pytest.raises(ValueError, match="unknown profile"):
        d.resolve_profile_name("bogus")


def test_build_compiler():
    c = d.build_compiler(max_tokens=1234, model="gpt-4o", core="CORE")
    assert c.max_tokens == 1234
    assert c.model_id == "gpt-4o"


def test_result_summary_full_and_empty():
    msgs = [{"role": "system"}, {"role": "user"}]
    r = SimpleNamespace(
        metadata={"token_count": "12", "tokens_budget": 100, "segments_kept": 2},
        messages=msgs,
        ref_map={1: "a"},
        ref_map_type={1: "t"},
        system_alerts=["x"],
    )
    s = d.result_summary(r)
    assert s["token_count"] == 12
    assert s["tokens_budget"] == 100
    assert s["tokens_trimmed"] == 0
    assert s["segments_offered"] == 0
    assert s["segments_kept"] == 2
    assert s["message_roles"] == ["system", "user"]
    assert s["ref_map"] == {1: "a"}
    assert s["system_alerts"] == ["x"]

    e = d.result_summary(
        SimpleNamespace(
            metadata=None, messages=None, ref_map={}, ref_map_type={}, system_alerts=[]
        )
    )
    assert e["token_count"] == 0
    assert e["tokens_budget"] is None
    assert e["message_roles"] == []
    assert e["metadata"] == {}


def test_print_helpers(capsys):
    from datetime import date

    d.print_json({"b": 1, "a": date(2020, 1, 2)})
    out = capsys.readouterr().out
    assert json.loads(out) == {"a": "2020-01-02", "b": 1}
    assert out.index('"a"') < out.index('"b"')
    d.print_lines(iter(["x", "y"]))
    assert capsys.readouterr().out == "x\ny\n"


def test_profile_descriptions_match_presets():
    from membrane.profile import PRESET_NAMES

    assert set(d.PROFILE_DESCRIPTIONS) == set(PRESET_NAMES)
