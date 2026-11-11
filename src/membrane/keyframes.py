"""Keyframes: the one sentence of a text that says the most, used to thin before dropping.

A keyframe is the densest sentence that fits a small token cap. Density is the mean rarity of
the sentence's terms, where a term's rarity is the inverse of how many sentences of a
reference set of texts it appears in. A sentence made of words that appear everywhere scores
low and one that carries rare terms scores high. Counting sentences, not texts, matters for
long texts: a phrase repeated on every line of one document is common, not rare. A sentence
over the cap can never be a keyframe; a text with no sentence under the cap has none.

A sentence that cannot stand alone stays with the one before it: one that opens with a
pronoun ("It ended at ...", "That was ...") or has at most three words. A keyframe is then
the sentence and such neighbours after it (up to three sentences), scored by its best
sentence, when together they fit the cap and the neighbour says enough: one with a pronoun
must be at least half as dense as the best sentence, a very short one must be the densest.
Otherwise the keyframe is the single sentence. Pure functions, no model calls.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from functools import lru_cache

DEFAULT_KEYFRAME_TOKENS = 40
THIN_MARK = " …"

# Words and identifiers: "call_12" and "ABC-123" stay whole, so a bare "12" does not match them.
_TERM = re.compile(r"[A-Za-z0-9]+(?:[-_'][A-Za-z0-9]+)*")
# Sentence ends, line breaks, and the " | " between fields of a serialized memory line.
_PIECE_END = re.compile(r"(?<=[.!?])\s+|\n+|\s+\|\s+")
# A serialized field that is only a label ("kind=note", "[fact] kind=note", "conf=1.00").
_LABEL_FIELD = re.compile(r"^(?:\[\w+\]\s*)?(?:\w+=\S*)?$")
_FIELD_KEY = re.compile(r"^\w+=")
# Openers that point back at the sentence before.
_BACK_REFERENCE = frozenset(
    ("it", "its", "it's", "they", "they're", "them", "their", "theirs", "this", "that",
     "that's", "these", "those", "he", "he's", "she", "she's", "him", "his", "her", "hers",
     "there", "there's", "then")
)
_OPENER, _SHORT = 1, 2
# A sentence with a back-reference comes along when it is at least this dense, relative to
# the best sentence of the keyframe. A very short one ("Not 4471.") only when it is the
# densest itself: "Understood." after a fact adds tokens and nothing else.
_LEAN_DENSITY = 0.5
_SHORT_SENTENCE_TERMS = 3
_SPAN_SENTENCES = 3


def terms(text: str) -> frozenset[str]:
    return frozenset(t.lower() for t in _TERM.findall(text))


@lru_cache(maxsize=8192)
def _split(text: str) -> tuple[tuple[str, frozenset[str], int], ...]:
    """Candidate sentences of a text in order: the sentence, its terms, and how it leans on
    the sentence before it in the same line: ``_OPENER`` when it opens with a back-reference,
    ``_SHORT`` when it is very short, else 0."""
    out: list[tuple[str, frozenset[str], int]] = []
    text = text.strip()
    start, follows = 0, False
    for end in [*_PIECE_END.finditer(text), None]:
        raw = text[start : end.start() if end else len(text)]
        piece = raw.strip()
        if piece.startswith("|"):  # a new field of a serialized line, not the next sentence
            piece, follows = piece.lstrip("|").strip(), False
        kept = False
        # Labels are not sentences, and neither is a fragment the serializer cut off.
        if piece and not _LABEL_FIELD.match(piece) and not piece.endswith("..."):
            piece = _FIELD_KEY.sub("", piece)
            found = terms(piece)
            if found:
                first = _TERM.search(piece)
                leans = 0
                if follows and first is not None and first.group(0).lower() in _BACK_REFERENCE:
                    leans = _OPENER
                elif follows and len(_TERM.findall(piece)) <= _SHORT_SENTENCE_TERMS:
                    leans = _SHORT
                out.append((piece, found, leans))
                kept = True
        if end is None:
            break
        # Only a plain sentence break joins two sentences; a line break or a field does not.
        follows = kept and end.group(0).isspace() and "\n" not in end.group(0)
        start = end.end()
    return tuple(out)


@lru_cache(maxsize=8192)
def _pieces(text: str) -> tuple[tuple[str, frozenset[str]], ...]:
    """Candidate sentences of a text with their term sets, in order."""
    return tuple((piece, found) for piece, found, _ in _split(text))


@lru_cache(maxsize=8192)
def _spans(text: str) -> tuple[tuple[int, int], ...]:
    """Candidate keyframes as (first sentence, one past the last): every sentence alone, and
    every run of a sentence with the leaning sentences after it, up to three sentences."""
    split = _split(text)
    spans = [(i, i + 1) for i in range(len(split))]
    for i in range(len(split)):
        for end in range(i + 2, min(len(split), i + _SPAN_SENTENCES) + 1):
            if not split[end - 1][2]:
                break
            spans.append((i, end))
    return tuple(spans)


def _pick(
    text: str,
    scores: list[float],
    count_fn: Callable[[str], int],
    cap: int,
    limit: int,
    density: list[float] | None = None,
) -> list[str]:
    """The best ``limit`` spans of ``text`` that fit ``cap``, in text order. ``scores`` has one
    score per sentence, lower is better; a span scores as its best sentence. Of equal spans
    the longer goes first (a leaning sentence comes along when it fits), then the earlier.
    With ``density`` (higher is denser), a span is only offered when its leaning sentences
    are dense enough (see ``_LEAN_DENSITY``) and no sentence in it is repeated."""
    split = _split(text)

    def offered(a: int, b: int) -> bool:
        if b - a == 1:
            return True
        if len({split[i][0] for i in range(a, b)}) < b - a:
            return False
        if density is None:  # matching a question: only a back-reference comes along
            return all(split[i][2] == _OPENER for i in range(a + 1, b))
        best = max(density[a:b])
        for i in range(a + 1, b):
            if split[i][2] == _SHORT and (density[i] < best or density[a:b].count(best) > 1):
                return False
            if density[i] < _LEAN_DENSITY * best:
                return False
        return True

    ranked = sorted(
        (min(scores[a:b]), a - b, min(range(a, b), key=lambda i: (scores[i], i)), a, b)
        for a, b in _spans(text)
        if offered(a, b)
    )
    used: set[int] = set()
    chosen: dict[str, int] = {}
    for score, _, _, a, b in ranked:
        if len(chosen) >= limit or score == math.inf:
            break
        if used.intersection(range(a, b)):
            continue
        frame = " ".join(split[i][0] for i in range(a, b))
        if frame not in chosen and count_fn(frame) <= cap:
            chosen[frame] = a
            used.update(range(a, b))
    return sorted(chosen, key=chosen.__getitem__)


@lru_cache(maxsize=8192)
def sentence_counts(text: str) -> tuple[tuple[tuple[str, int], ...], int]:
    """For one text: how many of its sentences each term appears in, and how many sentences."""
    counts: dict[str, int] = {}
    pieces = _pieces(text)
    for _, found in pieces:
        for term in found:
            counts[term] = counts.get(term, 0) + 1
    return tuple(counts.items()), len(pieces)


def add_sentences(df: dict[str, int], text: str) -> int:
    """Add one text's sentence counts to ``df``; returns how many sentences it had."""
    counts, n = sentence_counts(text)
    for term, count in counts:
        df[term] = df.get(term, 0) + count
    return n


def sentence_frequency(texts: Iterable[str]) -> tuple[dict[str, int], int]:
    """Over all the texts: how many sentences each term appears in, and how many sentences."""
    df: dict[str, int] = {}
    n = sum(add_sentences(df, text) for text in texts)
    return df, n


def keyframes(
    text: str,
    df: dict[str, int],
    n: int,
    count_fn: Callable[[str], int],
    cap: int = DEFAULT_KEYFRAME_TOKENS,
    limit: int = 1,
) -> list[str]:
    """The ``limit`` densest sentences of ``text`` that each fit ``cap`` tokens, in the order
    they appear in the text. A keyframe may be a sentence together with the sentences
    after it that lean on it (see the module text).

    ``df`` and ``n`` come from :func:`sentence_frequency`. Ties go to the earlier sentence.
    Terms missing from ``df`` count as seen once. A sentence repeated in the text counts once.
    """
    n = max(n, 1)
    scores = [
        -sum(math.log(1.0 + n / df.get(t, 1)) for t in found) / len(found)
        for _, found in _pieces(text)
    ]
    return _pick(text, scores, count_fn, cap, limit, [-score for score in scores])


def keyframe(
    text: str,
    df: dict[str, int],
    n: int,
    count_fn: Callable[[str], int],
    cap: int = DEFAULT_KEYFRAME_TOKENS,
) -> str | None:
    """The densest sentence of ``text`` that fits ``cap`` tokens, or ``None``."""
    found = keyframes(text, df, n, count_fn, cap, 1)
    return found[0] if found else None


def matching_sentence(
    text: str, query_terms: frozenset[str], count_fn: Callable[[str], int], cap: int
) -> str | None:
    """The sentence of ``text`` that shares the most with a query, or ``None`` when none
    shares anything or fits ``cap``. A shared term counts more the fewer of the text's own
    sentences have it. Ties go to the earlier sentence. Sentences after it that lean on it
    (they open with a pronoun or are very short) come along when they fit."""
    pieces = _pieces(text)
    counts = dict(sentence_counts(text)[0])
    total = max(len(pieces), 1)
    scores = [
        -sum(math.log(1.0 + total / counts[t]) for t in found & query_terms)
        for _, found in pieces
    ]
    # Only sentences that share something are offered; a leaning sentence comes with them.
    matching = [score if score < 0 else math.inf for score in scores]
    if all(score == math.inf for score in matching):
        return None
    found = _pick(text, matching, count_fn, cap, 1)
    return found[0] if found else None
