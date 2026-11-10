import pytest

from membrane import cache_format as cf


@pytest.fixture
def caching(monkeypatch):
    monkeypatch.setattr(cf, "has_prompt_caching", lambda m: m == "cached/model")


def test_make_cacheable_block(caching):
    b = cf.make_cacheable_block("cached/model", "pre", None)
    assert b == cf.CacheableBlock("pre", "", True)
    assert cf.make_cacheable_block("other", "pre", "t").use_caching is False
    assert cf.make_cacheable_block(None, "pre", "t").use_caching is False
    assert cf.make_cacheable_block("cached/model", "  ", "t").use_caching is False
    assert cf.make_cacheable_block("cached/model", None, "t").cacheable_prefix == ""


@pytest.mark.parametrize(
    ("prefix", "tail", "expected"),
    [
        ("", "tail", "tail"),
        ("  ", " tail", " tail"),
        ("pre", "", "pre"),
        ("pre", "  ", "pre"),
        ("pre\n\n", "\n tail", "pre\n\ntail"),
    ],
)
def test_assemble_system_content(prefix, tail, expected):
    assert cf.assemble_system_content(prefix, tail) == expected


def test_build_system_messages(caching):
    msgs, hint = cf.build_system_messages("cached/model", "pre", "tail")
    assert msgs == [{"role": "system", "content": "pre\n\ntail"}]
    assert hint is True
    msgs, hint = cf.build_system_messages("other", "pre", "tail")
    assert msgs[0]["content"] == "pre\n\ntail" and hint is False
    assert cf.build_system_messages("cached/model", "", "tail")[1] is False


def test_build_system_messages_empty(caching):
    assert cf.build_system_messages("cached/model", " ", "\n") == ([], False)


def test_anthropic_system_blocks():
    blocks = cf.anthropic_system_blocks("pre  \n", "  tail")
    assert blocks == [
        {"type": "text", "text": "pre", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "tail"},
    ]
    assert cf.anthropic_system_blocks("", "tail") == [{"type": "text", "text": "tail"}]
    assert cf.anthropic_system_blocks("", " ") == [{"type": "text", "text": ""}]


def test_is_anthropic_model():
    assert cf.is_anthropic_model(" Anthropic/claude ") is True
    assert cf.is_anthropic_model("openai/gpt") is False
    assert cf.is_anthropic_model(None) is False
    assert cf.is_anthropic_model("") is False


def test_real_registry_does_not_crash():
    msgs, hint = cf.build_system_messages("anthropic/claude-3-5-sonnet", "pre", "tail")
    assert msgs and isinstance(hint, bool)
