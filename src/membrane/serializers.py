"""Dense serialization of memory objects for prompt injection.

Structured lines use ``kind=… | key=value`` segments. Field values escape ``|`` as ``,`` and
collapse newlines so each memory stays one line. Episodic and procedural lines begin with
``[Mem:N]`` so layout and the preflight firewall can resolve ``[Mem:N]`` references.
"""

from __future__ import annotations

import contextlib
import logging
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from membrane import config as os_config
from membrane.memory_protocol import infer_memory_kind

logger = logging.getLogger(__name__)

EPISODIC_SUMMARY_MAX_CHARS = 280


# Set when a memory's main text is cut; read by ``DenseSerializer.format_checked``.
_text_truncated: ContextVar[bool] = ContextVar("membrane_text_truncated", default=False)


def _dense_text(value: Any, default_max_chars: int) -> str:
    """Main memory text: built-in limit unless ``MEMBRANE_MEMORY_TEXT_MAX_CHARS`` overrides it."""
    override = os_config.get_memory_text_max_chars()
    limit = default_max_chars if override is None else override
    if not limit:
        return _dense_field(value, None)
    out = _dense_field(value, limit)
    if out.endswith("...") and len(str(value).strip()) > limit:
        _text_truncated.set(True)
    return out


def _dense_field(value: Any, max_chars: int | None) -> str:
    """Normalize a value for dense key=value segments (pipe-safe, single line)."""
    s = str(value).replace("|", ",").replace("\n", " ").strip()
    if max_chars is not None and len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return s


class DenseSerializer:
    """Serializes semantic, episodic, and procedural memories into dense one-line strings."""

    @staticmethod
    def format(memory_object: Any, local_index: int | None = None) -> str:
        """Produce a single dense line for the memory object."""
        kind = infer_memory_kind(memory_object)
        if kind == "semantic":
            return DenseSerializer._format_fact(memory_object)
        if kind == "episodic":
            return DenseSerializer._format_episode(memory_object, local_index)
        if kind == "procedural":
            return DenseSerializer._format_procedure(memory_object, local_index)
        logger.warning("DenseSerializer.format received unknown type: %s", type(memory_object))
        return str(memory_object)[:EPISODIC_SUMMARY_MAX_CHARS]

    @staticmethod
    def _format_fact(fact: Any) -> str:
        content_only = getattr(fact, "content", None)
        subject = getattr(fact, "subject", "") or ""
        predicate = getattr(fact, "predicate", "") or ""
        if content_only and not str(subject).strip() and not str(predicate).strip():
            text = _dense_text(content_only, 200)
            return f"[fact] kind=note | text={text} | conf=1.00"
        obj = getattr(fact, "object", None)
        obj = getattr(fact, "object_value", "") or "" if obj is None else obj or ""
        confidence = getattr(fact, "confidence", 0.0)
        subj = _dense_text(subject, 200)
        rel = _dense_text(predicate, 200)
        o = _dense_text(obj, 200)
        try:
            conf_val = float(confidence)
        except (TypeError, ValueError):
            conf_val = 0.0
        return f"[fact] subj={subj} | rel={rel} | obj={o} | conf={conf_val:.2f}"

    @staticmethod
    def _format_episode(episode: Any, local_index: int | None) -> str:
        idx = local_index if local_index is not None else 0
        ts = getattr(episode, "timestamp", None)
        if isinstance(ts, datetime):
            date_str = ts.strftime("%Y-%m-%d")
        elif isinstance(ts, str):
            date_str = ts[:10] if len(ts) >= 10 else "0000-00-00"
        else:
            date_str = "0000-00-00"
        summary = (
            getattr(episode, "stage3_synthesis", None)
            or getattr(episode, "query", "")
            or getattr(episode, "content", "")
            or ""
        )
        text = _dense_text(summary, EPISODIC_SUMMARY_MAX_CHARS)
        return f"[Mem:{idx}] kind=episodic | date={date_str} | text={text}"

    @staticmethod
    def _format_procedure(procedure: Any, local_index: int | None) -> str:
        idx = local_index if local_index is not None else 0
        steps = getattr(procedure, "steps", []) or []
        text = _procedure_text(procedure)
        if not steps and text:
            # Plain procedural memories (e.g. KernelMemoryObject tier 1) carry their payload as
            # text.
            return f"[Mem:{idx}] kind=procedural | text={_dense_text(text, 500)}"
        name = getattr(procedure, "name", "") or "Unknown"
        name_safe = _dense_field(str(name).replace(":", "-"), 80)
        parts = []
        for s in steps:
            step_num = getattr(s, "step_number", len(parts) + 1)
            action = getattr(s, "action", None) or getattr(s, "description", "") or ""
            action_safe = str(action).replace("(", "").replace(")", "").replace(">", "-")[:100]
            parts.append(f"Step {step_num} ({action_safe})")
        steps_str = " > ".join(parts) if parts else "Step 1 (N/A)"
        steps_str = _dense_field(steps_str, 500)
        outcome = "N/A"
        if (
            parts
            and hasattr(steps[-1], "expected_outcome")
            and getattr(steps[-1], "expected_outcome", None)
        ):
            outcome = _dense_field(steps[-1].expected_outcome, 60)
        else:
            sr = getattr(procedure, "success_rate", None)
            if sr is not None:
                with contextlib.suppress(TypeError, ValueError):
                    outcome = f"{float(sr):.0%}"
        return (
            f"[Mem:{idx}] kind=procedural | name={name_safe} | steps={steps_str} "
            f"| outcome={outcome}"
        )

    @staticmethod
    def format_checked(memory_object: Any, local_index: int | None = None) -> tuple[str, bool]:
        """Like :meth:`format`, plus whether the memory's text was truncated."""
        token = _text_truncated.set(False)
        try:
            line = DenseSerializer.format(memory_object, local_index=local_index)
            return line, _text_truncated.get()
        finally:
            _text_truncated.reset(token)

    @staticmethod
    def format_full_prose(memory_object: Any) -> str:
        """Full prose for request_memory_expansion."""
        kind = infer_memory_kind(memory_object)
        if kind == "semantic":
            return DenseSerializer._format_fact_full_prose(memory_object)
        if kind == "episodic":
            return DenseSerializer._format_episode_full_prose(memory_object)
        if kind == "procedural":
            return DenseSerializer._format_procedure_full_prose(memory_object)
        logger.warning("DenseSerializer.format_full_prose unknown type: %s", type(memory_object))
        return str(memory_object)[:2000]

    @staticmethod
    def _format_fact_full_prose(fact: Any) -> str:
        subject = getattr(fact, "subject", "") or ""
        predicate = getattr(fact, "predicate", "") or ""
        obj = getattr(fact, "object", None)
        obj = getattr(fact, "object_value", "") or "" if obj is None else obj or ""
        confidence = getattr(fact, "confidence", None)
        parts = [str(subject), str(predicate), str(obj)]
        sentence = " ".join(p.strip() for p in parts if p and str(p).strip())
        if not sentence:
            return "(No content)"
        if confidence is not None:
            with contextlib.suppress(TypeError, ValueError):
                sentence += f" (confidence: {float(confidence):.2f})"
        return sentence + "."

    @staticmethod
    def _format_episode_full_prose(episode: Any) -> str:
        lines = []
        query = getattr(episode, "query", "") or ""
        synthesis = getattr(episode, "stage3_synthesis", "") or ""
        if query:
            lines.append(f"Query: {query.strip()}")
        if synthesis:
            lines.append(f"Synthesis: {synthesis.strip()}")
        ts = getattr(episode, "timestamp", None)
        if ts is not None:
            if isinstance(ts, datetime):
                lines.append(f"Date: {ts.strftime('%Y-%m-%d %H:%M')}")
            elif isinstance(ts, str):
                lines.append(f"Date: {ts[:16]}")
        domain = getattr(episode, "domain", None)
        if domain:
            lines.append(f"Domain: {domain}")
        insights = getattr(episode, "learned_insights", []) or []
        if insights:
            lines.append("Insights: " + "; ".join(str(i) for i in insights[:5]))
        return "\n".join(lines) if lines else "(No content)"

    @staticmethod
    def _format_procedure_full_prose(procedure: Any) -> str:
        text = _procedure_text(procedure)
        if not (getattr(procedure, "steps", None) or []) and text:
            return text
        lines = []
        name = getattr(procedure, "name", "") or "Unnamed procedure"
        desc = getattr(procedure, "description", "") or ""
        lines.append(f"Procedure: {name}")
        if desc:
            lines.append(f"Description: {desc}")
        steps = getattr(procedure, "steps", []) or []
        for i, s in enumerate(steps, start=1):
            step_num = getattr(s, "step_number", i)
            action = getattr(s, "action", None) or getattr(s, "description", "") or ""
            outcome = getattr(s, "expected_outcome", "") or ""
            lines.append(f"  Step {step_num}: {str(action).strip()}")
            if outcome:
                lines.append(f"    Expected outcome: {str(outcome).strip()}")
        return "\n".join(lines) if lines else "(No content)"

    @staticmethod
    def format_batch(
        memory_objects: list[Any],
        start_index: int = 1,
    ) -> tuple[list[str], dict[int, str]]:
        """Serialize list of objects with .id into lines and ref_map."""
        lines = []
        ref_map: dict[int, str] = {}
        for i, obj in enumerate(memory_objects):
            local_idx = start_index + i
            line = DenseSerializer.format(obj, local_index=local_idx)
            lines.append(line)
            global_id = getattr(obj, "id", None)
            if global_id is not None:
                ref_map[local_idx] = str(global_id)
        return lines, ref_map


def _procedure_text(procedure: Any) -> str:
    """Free-text body of a procedural memory without structured steps."""
    for attr in ("payload", "content"):
        val = getattr(procedure, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""
