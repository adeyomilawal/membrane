from membrane.firewall import count_tokens
from membrane.glaciator import GlaciateResult, glaciate


def test_empty_inputs():
    for val in (None, "", "  \n "):
        r = glaciate(val)
        assert r == GlaciateResult("", val or "", 0, 0, False)


def test_fits_entirely_in_prefix():
    text = "line one\nline two"
    r = glaciate(text, max_cacheable_tokens=1000)
    assert r.cacheable_prefix == text
    assert r.dynamic_tail == ""
    assert r.prefix_tokens == count_tokens(text)
    assert r.tail_tokens == 0
    assert r.use_caching is False  # under 256 tokens


def test_use_caching_threshold_when_fits():
    text = "\n".join(f"line number {i} with some words" for i in range(200))
    total = count_tokens(text)
    assert total >= 256
    r = glaciate(text, max_cacheable_tokens=total)
    assert r.use_caching is True and r.dynamic_tail == ""


def test_split_on_line_boundary():
    lines = [f"line number {i} with some words" for i in range(100)]
    text = "\n".join(lines)
    limit = count_tokens(text) // 2
    r = glaciate(text, max_cacheable_tokens=limit)
    assert r.cacheable_prefix and r.dynamic_tail
    assert r.prefix_tokens <= limit
    # only whole lines in each part, order preserved, nothing lost
    assert r.cacheable_prefix.split("\n") == lines[: len(r.cacheable_prefix.split("\n"))]
    assert r.cacheable_prefix + "\n" + r.dynamic_tail == text
    assert r.tail_tokens == count_tokens(r.dynamic_tail)


def test_first_line_exceeds_limit_goes_all_to_tail():
    text = "a very long first line " * 50 + "\nsecond"
    r = glaciate(text, max_cacheable_tokens=3)
    assert r.cacheable_prefix == ""
    assert r.dynamic_tail == text
    assert r.prefix_tokens == 0 and r.use_caching is False


def test_default_limit_from_env(monkeypatch):
    monkeypatch.setenv("MAX_CACHEABLE_PREFIX_TOKENS", "5")
    text = "\n".join(f"line number {i} with some words" for i in range(20))
    r = glaciate(text)
    assert r.dynamic_tail and r.prefix_tokens <= 5
