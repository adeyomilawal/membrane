"""Coverage tests for membrane.tokenizers.registry and adapters."""

import sys
import types

import pytest

from membrane.firewall import count_tokens
from membrane.tokenizers.adapters import (
    EstimationAdapter,
    HuggingFaceAdapter,
    SentencePieceAdapter,
    TiktokenAdapter,
)
from membrane.tokenizers.registry import TokenizerAdapter, TokenizerRegistry


class Fake:
    def __init__(self, label):
        self.label = label

    @property
    def name(self):
        return self.label

    def encode(self, text):
        return [len(text)]

    def decode(self, tokens):
        return f"{self.label}:{tokens}"

    def count(self, text):
        return 1000 + len(text)


@pytest.fixture(autouse=True)
def _reset_registry():
    TokenizerRegistry.reset_for_testing()
    yield
    TokenizerRegistry.reset_for_testing()


def _bare():
    return TokenizerRegistry()


def test_protocol_runtime_check():
    assert isinstance(Fake("x"), TokenizerAdapter)
    assert isinstance(EstimationAdapter(), TokenizerAdapter)
    assert not isinstance(object(), TokenizerAdapter)


def test_bare_registry_falls_back_to_firewall_count():
    r = _bare()
    assert r.count("anything", "hello world") == count_tokens("hello world")
    assert r.encode("anything", "hi") == []
    assert r.decode("anything", [1]) == ""
    assert r.list_models() == []


def test_register_precedence_and_case_insensitive():
    r = _bare()
    r.register("gpt-*", Fake("broad"))
    r.register("GPT-4o*", Fake("specific"))
    assert r.list_models() == ["GPT-4o*", "gpt-*"]
    assert r.decode("gpt-4O-mini", [1]) == "specific:[1]"
    assert r.decode("gpt-3", [2]) == "broad:[2]"
    assert r.count("gpt-3", "abc") == 1003
    assert r.encode("gpt-3", "abcd") == [4]


def test_unmatched_uses_default_adapter():
    r = _bare()
    r._default = EstimationAdapter(chars_per_token=2.0)
    assert r.count("unknown", "abcdef") == 3
    assert r.encode("unknown", "abcdef") == [0, 1, 2]
    assert r.decode("unknown", [1, 2]) == ""


def test_singleton_get_and_reset():
    a = TokenizerRegistry.get()
    assert TokenizerRegistry.get() is a
    TokenizerRegistry.reset_for_testing()
    assert TokenizerRegistry.get() is not a


def test_bootstrap_defaults_patterns_order():
    r = TokenizerRegistry.get()
    models = r.list_models()
    assert models.index("gpt-4o*") < models.index("gpt-4*")
    assert models.index("gpt-4.1*") < models.index("gpt-4*")
    assert "claude*" in models and "anthropic/*" in models
    # unknown model goes to estimation default
    assert r.count("mistral-7b", "a" * 35) == 10
    assert r.count("mistral-7b", "") == 0


def test_default_registry_real_tokenizers():
    r = TokenizerRegistry.get()
    assert r.count("gpt-4o", "hello world") == r.count("gpt-4.1-mini", "hello world")
    toks = r.encode("claude-3-opus", "hello world")
    assert toks and r.decode("claude-3-opus", toks) == "hello world"
    # user override beats defaults
    r.register("gpt-4o*", Fake("mine"))
    assert r.count("gpt-4o", "xy") == 1002


def test_tiktoken_adapter():
    a = TiktokenAdapter("cl100k_base")
    assert a.name == "tiktoken:cl100k_base"
    toks = a.encode("hello world")
    assert a.count("hello world") == len(toks)
    assert a.decode(toks) == "hello world"


def test_estimation_adapter():
    a = EstimationAdapter()
    assert a.name == "estimation"
    assert a.count("") == 0
    assert a.count("a") == 1  # floor clamps up to 1
    assert a.count("a" * 7) == 2
    assert a.encode("") == []
    assert a.encode("a" * 7) == [0, 1]
    assert a.decode([1, 2]) == ""


def _fake_sentencepiece(monkeypatch):
    class Proc:
        def load(self, path):
            self.path = path

        def encode_as_ids(self, text):
            return [ord(c) for c in text]

        def decode_ids(self, ids):
            return "".join(chr(i) for i in ids)

    mod = types.ModuleType("sentencepiece")
    mod.SentencePieceProcessor = Proc  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentencepiece", mod)


def test_sentencepiece_adapter_with_fake(monkeypatch):
    _fake_sentencepiece(monkeypatch)
    a = SentencePieceAdapter("/m.model")
    assert a.name == "sentencepiece:/m.model"
    assert a.encode("ab") == [97, 98]
    assert a.decode([97, 98]) == "ab"
    assert a.count("abc") == 3


def test_sentencepiece_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentencepiece", None)
    with pytest.raises(ImportError):
        SentencePieceAdapter("/m.model")


def test_huggingface_adapter_with_fake(monkeypatch):
    calls = {}

    class Tok:
        def encode(self, text, add_special_tokens=True):
            calls["enc"] = add_special_tokens
            return list(range(len(text.split())))

        def decode(self, tokens, skip_special_tokens=False):
            calls["dec"] = skip_special_tokens
            return "decoded"

    class Auto:
        @staticmethod
        def from_pretrained(model_id):
            calls["id"] = model_id
            return Tok()

    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = Auto  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", mod)

    a = HuggingFaceAdapter("org/model")
    assert a.name == "hf:org/model"
    assert calls["id"] == "org/model"
    assert a.count("one two three") == 3
    assert calls["enc"] is False
    assert a.decode([1]) == "decoded"
    assert calls["dec"] is True


def test_huggingface_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError):
        HuggingFaceAdapter("x")
