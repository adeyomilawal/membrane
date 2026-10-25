"""LangChain adapter (optional dep)."""

import pytest

from membrane.memory_protocol import MemoryItem

pytestmark = pytest.mark.requires_extras


def test_langchain_doc_to_memory_skips_without_langchain(monkeypatch):
    pytest.importorskip("langchain_core.documents", reason="langchain-core not installed")
    from langchain_core.documents import Document

    from membrane.adapters.langchain import langchain_doc_to_memory

    d = Document(page_content="hello", metadata={"id": "a1", "memory_type": "semantic"})
    m = langchain_doc_to_memory(d)
    assert isinstance(m, MemoryItem)
    assert m.content == "hello"
