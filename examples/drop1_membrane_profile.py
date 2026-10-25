"""
Drop 1 — MembraneProfile presets with ContextCompiler.

Run from repo root::

    PYTHONPATH=src python examples/drop1_membrane_profile.py
"""

from __future__ import annotations

from membrane.compiler import ContextCompiler
from membrane.memory_protocol import MemoryItem
from membrane.profile import MembraneProfile


def main() -> None:
    compiler = ContextCompiler(max_tokens=4096, model_id="gpt-4o-mini")
    profile = MembraneProfile.coding()
    memories = [
        MemoryItem(id="m1", content="Membrane packs context in a U-shape.", memory_type="semantic"),
    ]
    result = compiler.compile(memories=memories, user_message="Summarize Membrane.", profile=profile)
    print("messages:", len(result.messages))
    print("metadata keys:", list((result.metadata or {}).keys()))


if __name__ == "__main__":
    main()
