from datetime import datetime
from types import SimpleNamespace as NS

import pytest

from membrane import serializers
from membrane.serializers import DenseSerializer, _dense_field, _dense_text


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv("MEMBRANE_MEMORY_TEXT_MAX_CHARS", raising=False)


def fact(**kw):
    base = {"memory_kind": "semantic", "subject": "S", "predicate": "P", "object": "O",
            "confidence": 0.5}
    base.update(kw)
    return NS(**base)


def test_dense_field_escapes_and_truncates():
    assert _dense_field(" a|b\nc ", None) == "a,b c"
    assert _dense_field("x" * 20, 10) == "x" * 7 + "..."
    assert _dense_field("x" * 10, 10) == "x" * 10


def test_dense_text_override(monkeypatch):
    assert _dense_text("x" * 300, 200).endswith("...")
    monkeypatch.setenv("MEMBRANE_MEMORY_TEXT_MAX_CHARS", "0")
    assert _dense_text("x" * 300, 200) == "x" * 300
    monkeypatch.setenv("MEMBRANE_MEMORY_TEXT_MAX_CHARS", "20")
    assert len(_dense_text("x" * 300, 200)) == 20


def test_format_fact_structured():
    assert DenseSerializer.format(fact()) == "[fact] subj=S | rel=P | obj=O | conf=0.50"


def test_format_fact_object_value_fallback_and_bad_confidence():
    f = NS(memory_kind="semantic", subject="S", predicate="P", object=None,
           object_value="V", confidence="bad")
    assert DenseSerializer.format(f) == "[fact] subj=S | rel=P | obj=V | conf=0.00"


def test_format_fact_note():
    f = NS(memory_kind="semantic", content="hello | world", subject="", predicate="")
    assert DenseSerializer.format(f) == "[fact] kind=note | text=hello , world | conf=1.00"


def test_format_unknown_type_truncated():
    out = DenseSerializer.format(object())
    assert out.startswith("<object")
    assert len(DenseSerializer.format("z" * 1000)) == serializers.EPISODIC_SUMMARY_MAX_CHARS


def test_format_episode_dates():
    ep = NS(memory_kind="episodic", timestamp=datetime(2024, 3, 5, 10, 0), query="q",
            stage3_synthesis="")
    assert DenseSerializer.format(ep, 4) == "[Mem:4] kind=episodic | date=2024-03-05 | text=q"
    ep = NS(memory_kind="episodic", timestamp="2024-03-05T10:00", stage3_synthesis="syn")
    assert "date=2024-03-05 | text=syn" in DenseSerializer.format(ep, 1)
    ep = NS(memory_kind="episodic", timestamp="short", content="c")
    assert DenseSerializer.format(ep) == "[Mem:0] kind=episodic | date=0000-00-00 | text=c"
    ep = NS(memory_kind="episodic", timestamp=None, content="c")
    assert "date=0000-00-00" in DenseSerializer.format(ep, 2)


def test_format_procedure_text_only():
    p = NS(memory_kind="procedural", steps=[], payload="  do it  ")
    assert DenseSerializer.format(p, 3) == "[Mem:3] kind=procedural | text=do it"


def test_format_procedure_steps_with_outcome():
    steps = [
        NS(step_number=1, action="open (door)", expected_outcome=""),
        NS(step_number=2, description="walk > in", expected_outcome="inside"),
    ]
    p = NS(memory_kind="procedural", name="a:b", steps=steps)
    out = DenseSerializer.format(p, 7)
    assert out == (
        "[Mem:7] kind=procedural | name=a-b | "
        "steps=Step 1 (open door) > Step 2 (walk - in) | outcome=inside"
    )


def test_format_procedure_success_rate_fallback():
    p = NS(memory_kind="procedural", name="", steps=[NS(action="x")], success_rate=0.75)
    out = DenseSerializer.format(p, 1)
    assert "name=Unknown" in out and "outcome=75%" in out
    p = NS(memory_kind="procedural", name="n", steps=[NS(action="x")], success_rate="bad")
    assert DenseSerializer.format(p, 1).endswith("outcome=N/A")


def test_format_procedure_no_steps_no_text():
    p = NS(memory_kind="procedural", name="n", steps=[])
    assert DenseSerializer.format(p, 1).endswith("steps=Step 1 (N/A) | outcome=N/A")


def test_format_checked_flags_truncation():
    line, trunc = DenseSerializer.format_checked(fact(object="x" * 500))
    assert trunc and "..." in line
    line, trunc = DenseSerializer.format_checked(fact())
    assert not trunc
    # flag does not leak between calls
    assert DenseSerializer.format_checked(fact())[1] is False


def test_full_prose_fact():
    assert DenseSerializer.format_full_prose(fact()) == "S P O (confidence: 0.50)."
    empty = NS(memory_kind="semantic", subject="", predicate="", object="")
    assert DenseSerializer.format_full_prose(empty) == "(No content)"
    odd = fact(confidence="bad")
    assert DenseSerializer.format_full_prose(odd) == "S P O."


def test_full_prose_episode():
    ep = NS(memory_kind="episodic", query=" q ", stage3_synthesis=" s ",
            timestamp=datetime(2024, 1, 2, 3, 4), domain="d",
            learned_insights=[str(i) for i in range(7)])
    assert DenseSerializer.format_full_prose(ep).split("\n") == [
        "Query: q", "Synthesis: s", "Date: 2024-01-02 03:04", "Domain: d",
        "Insights: 0; 1; 2; 3; 4",
    ]
    ep = NS(memory_kind="episodic", query="", stage3_synthesis="", timestamp="2024-01-02T03:04:05")
    assert DenseSerializer.format_full_prose(ep) == "Date: 2024-01-02T03:04"
    assert DenseSerializer.format_full_prose(NS(memory_kind="episodic")) == "(No content)"


def test_full_prose_procedure():
    assert DenseSerializer.format_full_prose(
        NS(memory_kind="procedural", steps=[], content=" body ")) == "body"
    steps = [
        NS(step_number=1, action=" a ", expected_outcome=" ok "),
        NS(step_number=3, description="b"),
    ]
    p = NS(memory_kind="procedural", name="N", description="D", steps=steps)
    assert DenseSerializer.format_full_prose(p).split("\n") == [
        "Procedure: N", "Description: D", "  Step 1: a", "    Expected outcome: ok",
        "  Step 3: b",
    ]
    p = NS(memory_kind="procedural", steps=[])
    assert DenseSerializer.format_full_prose(p) == "Procedure: Unnamed procedure"


def test_full_prose_unknown():
    assert DenseSerializer.format_full_prose("y" * 3000) == "y" * 2000


def test_format_batch():
    a = fact(id="g1")
    b = NS(memory_kind="episodic", query="q")
    lines, ref_map = DenseSerializer.format_batch([a, b], start_index=5)
    assert lines[0].startswith("[fact]") and lines[1].startswith("[Mem:6]")
    assert ref_map == {5: "g1"}


def test_full_prose_numbers_unnumbered_steps_by_position() -> None:
    from types import SimpleNamespace

    from membrane.serializers import DenseSerializer

    procedure = SimpleNamespace(
        name="Deploy",
        description="Ship a release",
        steps=[
            SimpleNamespace(action="Build", expected_outcome="Wheel exists"),
            SimpleNamespace(action="Upload", expected_outcome=""),
        ],
    )
    prose = DenseSerializer._format_procedure_full_prose(procedure)
    assert "Step 1: Build" in prose
    assert "Step 2: Upload" in prose
