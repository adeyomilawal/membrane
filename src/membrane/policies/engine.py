"""Policy engine: evaluate conditions and return compiler kw overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence



@dataclass
class PolicyCondition:
    """Serializable condition descriptor."""

    raw: dict[str, Any]

    def matches(
        self,
        *,
        query: str,
        memories: Sequence[Any],
        estimated_tokens: int,
        model_id: str | None,
    ) -> bool:
        d = self.raw
        if d is True or d == "always":
            return True
        if not isinstance(d, dict):
            return False
        if "always" in d and not d["always"]:
            return False
        q = query.lower()
        mid = model_id or ""
        # every recognized key must hold (AND)
        if "query_contains_any" in d and not any(
            str(n).lower() in q for n in d["query_contains_any"]
        ):
            return False
        if "query_contains_all" in d and not all(
            str(n).lower() in q for n in d["query_contains_all"]
        ):
            return False
        if "query_length_below" in d and not len(query) < int(d["query_length_below"]):
            return False
        if "query_length_above" in d and not len(query) > int(d["query_length_above"]):
            return False
        if "memory_count_below" in d and not len(memories) < int(d["memory_count_below"]):
            return False
        if "memory_count_above" in d and not len(memories) > int(d["memory_count_above"]):
            return False
        if "estimated_tokens_above" in d and not estimated_tokens > int(
            d["estimated_tokens_above"]
        ):
            return False
        if "estimated_tokens_below" in d and not estimated_tokens < int(
            d["estimated_tokens_below"]
        ):
            return False
        return not ("model_is" in d and not (mid == d["model_is"] or mid.startswith(d["model_is"])))


_LIST_KEYS = ("query_contains_any", "query_contains_all")
_INT_KEYS = (
    "query_length_below",
    "query_length_above",
    "memory_count_below",
    "memory_count_above",
    "estimated_tokens_above",
    "estimated_tokens_below",
)
_CONDITION_KEYS = frozenset(("always", "model_is", *_LIST_KEYS, *_INT_KEYS))


_ACTION_KEYS = frozenset(
    (
        "set_budget",
        "set_max_tokens",
        "set_compression_target",
        "pin_memory_kind",
        "enable",
        "disable",
        "override",
    )
)
_MEMORY_KINDS = frozenset(("procedural", "semantic", "episodic"))


def _validate_action(policy_name: str, action: Any) -> None:
    """Raise ``ValueError`` for non-dict actions, unknown action keys, or unknown memory kinds."""
    if not isinstance(action, dict) or not action:
        raise ValueError(f"policy {policy_name!r}: malformed action {action!r}")
    unknown = sorted(set(action) - _ACTION_KEYS)
    if unknown:
        raise ValueError(f"policy {policy_name!r}: unrecognized action keys {unknown}")
    kinds = list((action.get("set_budget") or {}).keys())
    if "pin_memory_kind" in action:
        kinds.append(action["pin_memory_kind"])
    bad = sorted(str(k) for k in kinds if k not in _MEMORY_KINDS)
    if bad:
        raise ValueError(f"policy {policy_name!r}: action names unknown memory kinds {bad}")
    if "override" in action and not isinstance(action["override"], dict):
        raise ValueError(f"policy {policy_name!r}: action 'override' must be a mapping")


def _validate_condition(policy_name: str, cond: dict[str, Any]) -> None:
    """Raise ``ValueError`` for empty, unknown-key, or malformed conditions."""
    unknown = sorted(set(cond) - _CONDITION_KEYS)
    if unknown or not cond:
        raise ValueError(
            f"policy {policy_name!r}: unrecognized condition keys {unknown or list(cond)}"
        )
    for k in _LIST_KEYS:
        if k in cond and (not isinstance(cond[k], (list, tuple)) or not cond[k]):
            raise ValueError(f"policy {policy_name!r}: {k} must be a non-empty list")
    for k in _INT_KEYS:
        if k in cond:
            try:
                int(cond[k])
            except (TypeError, ValueError) as e:
                raise ValueError(f"policy {policy_name!r}: {k} must be an integer") from e
    if "model_is" in cond and not isinstance(cond["model_is"], str):
        raise ValueError(f"policy {policy_name!r}: model_is must be a string")


@dataclass
class PolicyAction:
    raw: dict[str, Any]


@dataclass
class Policy:
    name: str
    condition: PolicyCondition
    actions: list[PolicyAction]
    priority: int = 0


class PolicyEngine:
    """Loads policies from dict/YAML and produces compiler kwargs overrides."""

    def __init__(self, policies: list[Policy]) -> None:
        self._policies = sorted(policies, key=lambda p: -p.priority)

    @classmethod
    def from_yaml(cls, path: str) -> PolicyEngine:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError("PolicyEngine.from_yaml requires PyYAML") from e
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data if isinstance(data, dict) else {})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyEngine:
        raw_list = data.get("policies") or []
        policies: list[Policy] = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "anonymous"))
            if "condition" not in item:
                raise ValueError(
                    f"policy {name!r}: missing condition (use \"always\" to match every request)"
                )
            cr = item.get("condition")
            if cr == "always" or cr is True:
                cond_raw: Any = {"always": True}
            elif isinstance(cr, dict):
                _validate_condition(name, cr)
                cond_raw = cr
            else:
                raise ValueError(f"policy {name!r}: malformed condition {cr!r}")
            cond = PolicyCondition(raw=cond_raw)
            actions_raw = item.get("actions") or []
            for a in actions_raw:
                _validate_action(name, a)
            actions = [PolicyAction(raw=a) for a in actions_raw]
            pr = int(item.get("priority", 0))
            policies.append(Policy(name=name, condition=cond, actions=actions, priority=pr))
        return cls(policies)

    def apply(
        self,
        *,
        query: str,
        memories: Sequence[Any],
        compiler_kwargs: dict[str, Any],
        estimated_tokens: int = 0,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        out = dict(compiler_kwargs)
        for pol in self._policies:
            if not pol.condition.matches(
                query=query,
                memories=memories,
                estimated_tokens=estimated_tokens,
                model_id=model_id,
            ):
                continue
            for act in pol.actions:
                self._apply_action(out, act.raw)
        return out

    def _apply_action(self, out: dict[str, Any], raw: dict[str, Any]) -> None:
        if "set_budget" in raw:
            b = raw["set_budget"]
            out.setdefault("policy_budget_override", {}).update(b)
        if "set_max_tokens" in raw:
            out["policy_max_tokens"] = int(raw["set_max_tokens"])
        if "set_compression_target" in raw:
            out["policy_compression_target"] = float(raw["set_compression_target"])
        if "pin_memory_kind" in raw:
            out.setdefault("policy_pin_kinds", []).append(raw["pin_memory_kind"])
        if "enable" in raw:
            for feat in raw["enable"]:
                out.setdefault("policy_enable", []).append(feat)
        if "disable" in raw:
            for feat in raw["disable"]:
                out.setdefault("policy_disable", []).append(feat)
        if "override" in raw:
            out.update(raw["override"])
