"""Keyframes: the one sentence of a text that says the most, used to thin before dropping.

A keyframe is the densest sentence that fits a small token cap. Density is the mean rarity of
the sentence's terms, where a term's rarity is the inverse of how many sentences of a
reference set of texts it appears in. A sentence made of words that appear everywhere scores
low and one that carries rare terms scores high. Counting sentences, not texts, matters for
long texts: a phrase repeated on every line of one document is common, not rare. A sentence
over the cap can never be a keyframe; a text with no sentence under the cap has none. Pure
functions, no model calls.
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


def terms(text: str) -> frozenset[str]:
    return frozenset(t.lower() for t in _TERM.findall(text))


@lru_cache(maxsize=8192)
def _pieces(text: str) -> tuple[tuple[str, frozenset[str]], ...]:
    """Candidate sentences of a text with their term sets, in order."""
    out: list[tuple[str, frozenset[str]]] = []
    for raw in _PIECE_END.split(text.strip()):
        piece = raw.strip().lstrip("|").strip()
        # Labels are not sentences, and neither is a fragment the serializer cut off.
        if not piece or _LABEL_FIELD.match(piece) or piece.endswith("..."):
            continue
        piece = _FIELD_KEY.sub("", piece)
        found = terms(piece)
        if found:
            out.append((piece, found))
    return tuple(out)


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
    they appear in the text.

    ``df`` and ``n`` come from :func:`sentence_frequency`. Ties go to the earlier sentence.
    Terms missing from ``df`` count as seen once. A sentence repeated in the text counts once.
    """
    n = max(n, 1)
    scored = [
        (-sum(math.log(1.0 + n / df.get(t, 1)) for t in found) / len(found), position, piece)
        for position, (piece, found) in enumerate(_pieces(text))
    ]
    chosen: dict[str, int] = {}
    for _, position, piece in sorted(scored):
        if len(chosen) >= limit:
            break
        if piece not in chosen and count_fn(piece) <= cap:
            chosen[piece] = position
    return sorted(chosen, key=chosen.__getitem__)


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
    sentences have it. Ties go to the earlier sentence."""
    pieces = _pieces(text)
    counts = dict(sentence_counts(text)[0])
    total = max(len(pieces), 1)
    scored = [
        (-sum(math.log(1.0 + total / counts[t]) for t in found & query_terms), position, piece)
        for position, (piece, found) in enumerate(pieces)
    ]
    for score, _, piece in sorted(scored):
        if score < 0 and count_fn(piece) <= cap:
            return piece
    return None
