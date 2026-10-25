"""Illustrates injecting retrieved rows as MemoryItem then compiling."""

from membrane import ContextCompiler, MemoryItem


def fake_retrieve(_query: str) -> list[MemoryItem]:
    return [
        MemoryItem(id="d1", content="Doc chunk about asyncio.", memory_type="semantic", priority=2),
    ]


def main() -> None:
    q = "How do I await?"
    memories = fake_retrieve(q)
    compiler = ContextCompiler(max_tokens=8192, core_content="Answer using context only.")
    result = compiler.compile(memories=memories, user_message=q)
    print(result.metadata.get("retrieval_summary"))


if __name__ == "__main__":
    main()
