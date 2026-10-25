"""Editable install: PYTHONPATH=src python examples/basic_compile.py"""

from membrane import CompileResult, ContextCompiler, MemoryItem


def main() -> None:
    compiler = ContextCompiler(max_tokens=4096, model_id="openai/gpt-4o", core_content="You are helpful.")
    memories = [
        MemoryItem(id="a", content="User likes Python.", memory_type="semantic", priority=2),
    ]
    result: CompileResult = compiler.compile(memories=memories, user_message="Hello")
    print(result.messages[0]["role"], len(result.messages[0].get("content", "")))
    print("metadata", result.metadata)


if __name__ == "__main__":
    main()
