"""U-shaped prompt assembly: core (start), sacrifice zone (middle), end block."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from membrane.memory_protocol import infer_memory_kind
from membrane.serializers import DenseSerializer

logger = logging.getLogger(__name__)


class ContextLayoutEngine:
    """Builds U-shaped prompt: core (start), sacrifice zone (middle), last turn + user (end)."""

    def __init__(
        self,
        core_content: Optional[str] = "",
        system_directives: Optional[str] = None,
    ):
        self.core_content = core_content or ""
        self.system_directives = system_directives or ""
        # Memories whose text was shortened by the last build_middle() call.
        self.last_truncated_count = 0

    def set_system_directives(self, directives: str) -> None:
        """Inject dynamic directives (e.g. metacognition) for end-of-system placement."""
        self.system_directives = directives

    def build_core(self) -> str:
        """Assemble start block: core content only (directives go via build_directives_block)."""
        return (self.core_content or "").strip()

    def build_directives_block(self) -> str:
        """ACTIVE_SYSTEM_DIRECTIVES XML block for appending at bottom of system message."""
        if not self.system_directives or not self.system_directives.strip():
            return ""
        return (
            "\n\n<ACTIVE_SYSTEM_DIRECTIVES>\n"
            + self.system_directives.strip()
            + "\n</ACTIVE_SYSTEM_DIRECTIVES>"
        )

    def build_middle(
        self,
        retrieved: Sequence[Tuple[Any, int]],
        start_index: int = 1,
    ) -> Tuple[List[Tuple[str, int]], Dict[int, str], Dict[int, str]]:
        """
        Serialize retrieved memories with local [Mem:N] and tier.

        Returns:
            (middle_segments, ref_map, ref_map_type).
        """
        segments: List[Tuple[str, int]] = []
        ref_map: Dict[int, str] = {}
        ref_map_type: Dict[int, str] = {}
        self.last_truncated_count = 0
        for i, (obj, tier) in enumerate(retrieved):
            local_idx = start_index + i
            line, truncated = DenseSerializer.format_checked(obj, local_index=local_idx)
            self.last_truncated_count += int(truncated)
            kind = infer_memory_kind(obj)
            # Episodic/procedural already start with [Mem:N]; semantic [fact] lines get the prefix here
            if kind == "semantic" and not line.strip().startswith("[Mem:"):
                line = f"[Mem:{local_idx}] {line}"
            segments.append((line, tier))
            global_id = getattr(obj, "id", None)
            if global_id is not None:
                ref_map[local_idx] = str(global_id)
                mk = infer_memory_kind(obj)
                if mk:
                    ref_map_type[local_idx] = mk
        return segments, ref_map, ref_map_type

    @staticmethod
    def build_end(
        last_turn_content: Optional[str],
        user_message: str,
    ) -> str:
        """Assemble end block: last turn (if any) + current user message."""
        parts = []
        if last_turn_content and last_turn_content.strip():
            parts.append("--- Last turn ---")
            parts.append(last_turn_content.strip())
            parts.append("")
        parts.append("--- Current user message ---")
        parts.append(user_message.strip())
        return "\n".join(parts)

    def assemble(
        self,
        core_text: Optional[str] = None,
        middle_segments: Optional[Sequence[Tuple[str, int]]] = None,
        end_text: Optional[str] = None,
        ref_map: Optional[Dict[int, str]] = None,
    ) -> Tuple[str, Dict[int, str]]:
        """Concatenate core + middle + end (no token cap; use PreFlightContextFirewall)."""
        core = (core_text if core_text is not None else self.build_core()) or ""
        middle_list = list(middle_segments or [])
        middle_str = "\n".join(s[0] for s in middle_list)
        end = end_text or ""
        full = core + "\n\n" + middle_str + "\n\n" + end
        return full, dict(ref_map or {})
