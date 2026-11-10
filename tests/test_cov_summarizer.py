from membrane.conversation.summarizer import (
    extractive_summary,
    score_sentences,
    summarize_turns,
)


def test_score_sentences_orders_by_length_and_digit_bonus():
    scored = score_sentences("Short one. This sentence is clearly longer. Mid 2 words.")
    assert scored[0][0] == "This sentence is clearly longer."
    d = dict(scored)
    assert d["Short one."] == 2.0
    assert d["Mid 2 words."] == 3.15


def test_score_sentences_empty():
    assert score_sentences("") == []
    assert score_sentences("   ") == []


def test_digit_bonus_breaks_tie():
    scored = score_sentences("aaa bbb. ccc 1.")
    assert scored[0][0] == "ccc 1."


def test_extractive_summary_limits():
    text = "One two three. Four five. Six."
    assert extractive_summary(text, max_sentences=1) == "One two three."
    assert extractive_summary(text, max_sentences=2) == "One two three. Four five."
    # char cap stops at first sentence that does not fit
    assert extractive_summary(text, max_chars=15) == "One two three."


def test_extractive_summary_fallback_slice():
    assert extractive_summary("Hello world.", max_chars=5) == "Hello"
    assert extractive_summary("") == ""


def test_summarize_turns_custom_fn():
    seen = []

    def fn(blob):
        seen.append(blob)
        return "S"

    assert summarize_turns([("user", "hi"), ("assistant", "yo")], fn) == "S"
    assert seen == ["user: hi\nassistant: yo"]


def test_summarize_turns_extractive_default():
    out = summarize_turns([("user", "Hello there friend."), ("assistant", "Hi.")], None)
    assert out.startswith("user: Hello there friend.")
    assert summarize_turns([], None) == ""
