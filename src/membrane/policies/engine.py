"""Policy engine: evaluate conditions and return compiler kw overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence



@dataclass
class PolicyCondition:
    """Serializable condition descriptor."""

    raw: Dict[str, Any]

    def matches(
        self,
        *,
        query: str,
        memories: Sequence[Any],
        estimated_tokens: int,
        model_id: Optional[str],
    ) -> bool:
        d = self.raw
        if d is True or d == "always" or (isinstance(d, dict) and d.get("always")):
            return True
        if not isinstance(d, dict):
            return False
        if "query_contains_any" in d:
            needles = d["query_contains_any"]
            q = query.lower()
            return any(str(n).lower() in q for n in needles)
        if "query_contains_all" in d:
            needles = d["query_contains_all"]
            q = query.lower()
            return all(str(n).lower() in q for n in needles)
        if "query_length_below" in d:
            return len(query) < int(d["query_length_below"])
        if "query_length_above" in d:
            return len(query) > int(d["query_length_above"])
        if "memory_count_below" in d:
            return len(memories) < int(d["memory_count_below"])
        if "memory_count_above" in d:
            return len(memories) > int(d["memory_count_above"])
        if "estimated_tokens_above" in d:
            return estimated_tokens > int(d["estimated_tokens_above"])
        if "estimated_tokens_below" in d:
            return estimated_tokens < int(d["estimated_tokens_below"])
        if "model_is" in d:
            mid = d["model_is"]
            return (model_id or "") == mid or (model_id or "").startswith(mid)
        return False


@dataclass
class PolicyAction:
    raw: Dict[str, Any]


@dataclass
class Policy:
    name: str
    condition: PolicyCondition
    actions: List[PolicyAction]
    priority: int = 0


class PolicyEngine:
    """Loads policies from dict/YAML and produces compiler kwargs overrides."""

    def __init__(self, policies: List[Policy]) -> None:
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
    def from_dict(cls, data: Dict[str, Any]) -> PolicyEngine:
        raw_list = data.get("policies") or []
        policies: List[Policy] = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "anonymous"))
            cr = item.get("condition")
            if cr == "always" or cr is True:
                cond_raw: Any = {"always": True}
            elif isinstance(cr, dict):
                cond_raw = cr
            else:
                cond_raw = {"always": True}
            cond = PolicyCondition(raw=cond_raw)
            actions_raw = item.get("actions") or []
            actions = [PolicyAction(raw=a) if isinstance(a, dict) else PolicyAction(raw={}) for a in actions_raw]
            pr = int(item.get("priority", 0))
            policies.append(Policy(name=name, condition=cond, actions=actions, priority=pr))
        return cls(policies)

    def apply(
        self,
        *,
        query: str,
        memories: Sequence[Any],
        compiler_kwargs: Dict[str, Any],
        estimated_tokens: int = 0,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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

    def _apply_action(self, out: Dict[str, Any], raw: Dict[str, Any]) -> None:
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
