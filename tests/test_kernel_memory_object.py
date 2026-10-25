"""KernelMemoryObject (0A), schema loader, bridges, and compile normalization."""

from __future__ import annotations

from membrane import (
    CompileResult,
    ContextCompiler,
    KernelMemoryObject,
    MemoryItem,
    kernel_memory_to_memory_item,
    load_kernel_memory_json_schema,
    memory_item_to_kernel_memory,
    normalize_compile_memories,
)


def test_kernel_memory_to_dict_roundtrip():
    k = KernelMemoryObject(
        id="m1",
        payload="hello",
        relevance_score=0.9,
        semantic_tier=3,
        temporal_weight=0.5,
        provenance={"src": "test"},
    )
    d = k.to_dict()
    k2 = KernelMemoryObject.from_dict(d)
    assert k2.id == k.id
    assert k2.payload == k.payload
    assert k2.semantic_tier == 3
    assert k2.memory_kind == "episodic"
    assert k2.priority == 3
    assert k2.content == "hello"


def test_from_dict_clamps_semantic_tier():
    k = KernelMemoryObject.from_dict({"payload": "x", "semantic_tier": 99})
    assert k.semantic_tier == 3


def test_from_dict_non_string_payload():
    k = KernelMemoryObject.from_dict({"payload": {"a": 1}, "semantic_tier": 1})
    assert '"a"' in k.payload or "1" in k.payload
    assert k.semantic_tier == 1
    assert k.memory_kind == "procedural"


def test_memory_item_bridge_roundtrip():
    mi = MemoryItem(id="u1", content="fact", memory_type="semantic", priority=2)
    k = memory_item_to_kernel_memory(mi)
    assert k.payload == "fact"
    assert k.semantic_tier == 2
    mi2 = kernel_memory_to_memory_item(k)
    assert mi2.id == mi.id
    assert mi2.content == mi.content
    assert mi2.priority == mi.priority


def test_normalize_compile_memories_dict():
    raw = [{"id": "d1", "payload": "from json", "semantic_tier": 2}]
    out = normalize_compile_memories(raw)
    assert len(out) == 1
    assert isinstance(out[0], KernelMemoryObject)
    assert out[0].payload == "from json"


def test_load_kernel_memory_json_schema():
    schema = load_kernel_memory_json_schema()
    assert schema["title"] == "KernelMemoryObject"
    assert "payload" in schema.get("properties", {})


def test_compile_accepts_kernel_dict_rows():
    compiler = ContextCompiler(
        max_tokens=8192,
        model_id="openai/gpt-4o",
        core_content="You are helpful.",
    )
    memories = [
        {"id": "k1", "payload": "User likes tests.", "semantic_tier": 2},
        MemoryItem(id="m1", content="Legacy row.", memory_type="semantic", priority=2),
    ]
    result: CompileResult = compiler.compile(
        memories=memories,
        user_message="Hi",
    )
    assert isinstance(result, CompileResult)
    assert result.messages
    full = "\n".join(str(m.get("content", "")) for m in result.messages)
    assert "User likes tests." in full or "Legacy row." in full
