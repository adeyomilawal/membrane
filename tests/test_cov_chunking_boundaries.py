import pytest

from membrane.chunking.boundaries import (
    _cosine_dist,
    detect_headings,
    detect_paragraphs,
    detect_sentences,
    detect_topic_shifts,
)


def test_sentences_spans_slice_back():
    text = "Hello there. How are you? Fine!"
    spans = detect_sentences(text)
    assert [text[a:b] for a, b in spans] == ["Hello there.", "How are you?", "Fine!"]


def test_sentences_empty_and_whitespace():
    assert detect_sentences("") == []
    assert detect_sentences("   \n") == []


def test_sentences_leading_whitespace_offsets():
    text = "  One. Two."
    assert [text[a:b] for a, b in detect_sentences(text)] == ["One.", "Two."]


def test_paragraphs():
    text = "a\n\nbb\n\n\n\nccc"
    assert [text[a:b] for a, b in detect_paragraphs(text)] == ["a", "bb", "ccc"]
    assert detect_paragraphs("") == [(0, 0)]
    assert detect_paragraphs("single") == [(0, 6)]


def test_headings():
    text = "# T\nbody\n## Sub heading\n####### seven\n#nospace"
    hs = detect_headings(text)
    assert [(text[a:b], lvl) for a, b, lvl in hs] == [("# T", 1), ("## Sub heading", 2)]


def test_topic_shifts_without_embedder_uses_paragraphs():
    text = "a\n\nb\n\nc"
    assert detect_topic_shifts(text) == [3, 6]
    assert detect_topic_shifts("one paragraph") == []


def test_topic_shifts_with_embedder():
    text = "Cats purr. Cats meow. Stocks fell."

    def embed(s):
        return [1.0, 0.0] if "Cats" in s else [0.0, 1.0]

    assert detect_topic_shifts(text, embed) == [text.index("Stocks")]
    assert detect_topic_shifts(text, embed, threshold=2.0) == []


def test_topic_shifts_single_sentence():
    assert detect_topic_shifts("Only one.", lambda s: [1.0]) == []


def test_cosine_dist():
    assert _cosine_dist([1, 0], [1, 0]) == pytest.approx(0.0)
    assert _cosine_dist([1, 0], [0, 1]) == pytest.approx(1.0)
    assert _cosine_dist([1, 0], [-1, 0]) == pytest.approx(2.0)
    assert _cosine_dist([0, 0], [1, 1]) == 0.0
