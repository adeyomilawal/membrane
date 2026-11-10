"""Keyframes: the one sentence of a text that says the most, used to thin before dropping.

A keyframe is the densest sentence that fits a small token cap. Density is the mean inverse
document frequency of the sentence's terms over a reference set of texts, so a sentence made
of words that appear everywhere scores low and one that carries the text's rare terms scores
high. A sentence over the cap can never be a keyframe; a text with no sentence under the cap
has none. Pure functions, no model calls.
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
def cached_terms(text: str) -> frozenset[str]:
    return terms(text)


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


def document_frequency(term_sets: Iterable[frozenset[str]]) -> tuple[dict[str, int], int]:
    """How many of the texts each term appears in, and the number of texts."""
    df: dict[str, int] = {}
    n = 0
    for found in term_sets:
        n += 1
        for term in found:
            df[term] = df.get(term, 0) + 1
    return df, n


def keyframe(
    text: str,
    df: dict[str, int],
    n: int,
    count_fn: Callable[[str], int],
    cap: int = DEFAULT_KEYFRAME_TOKENS,
) -> str | None:
    """The densest sentence of ``text`` that fits ``cap`` tokens, or ``None``.

    Ties go to the earlier sentence. Terms missing from ``df`` count as seen once.
    """
    scored = [
        (-sum(math.log(1.0 + n / df.get(t, 1)) for t in found) / len(found), position, piece)
        for position, (piece, found) in enumerate(_pieces(text))
    ]
    for _, _, piece in sorted(scored):
        if count_fn(piece) <= cap:
            return piece
    return None
