"""Multi-turn: pass conversation_messages for LOD + last-turn end block."""

from membrane import ContextCompiler, MemoryItem


def main() -> None:
    compiler = ContextCompiler(max_tokens=4096, core_content="You are a chatbot.")
    messages = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "A programming language."},
    ]
    result = compiler.compile(
        memories=[MemoryItem(id="1", content="User learns fast.", memory_type="episodic", priority=3)],
        user_message="Show a list comprehension.",
        conversation_messages=messages,
    )
    print("attention_shift", result.metadata.get("attention_shift"))


if __name__ == "__main__":
    main()
