"""Templates and chunking."""

from membrane.chunking import ChunkingEngine
from membrane.chunking.strategies import FixedStrategy
from membrane.templates import PromptTemplate


def test_prompt_template_render():
    t = PromptTemplate("{% system %}X{{ name }}{% end_system %}")
    msgs = t.render(name="bob")
    assert any("bob" in str(m.get("content", "")) for m in msgs)


def test_template_extends_block():
    parent = "{% block greeting %}P{% endblock %}"
    child = '{% extends "base" %}{% block greeting %}Hello{% endblock %}'
    t = PromptTemplate(child, loader={"base": parent})
    msgs = t.render()
    text = "\n".join(m.get("content", "") for m in msgs)
    assert "Hello" in text
    assert "P" not in text


def test_chunking_fixed():
    eng = ChunkingEngine(FixedStrategy())
    chunks = eng.chunk("a " * 100, max_tokens=10)
    assert chunks
