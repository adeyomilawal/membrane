"""A keyframe may be a sentence together with the sentences after it that lean on it."""

from __future__ import annotations

from membrane.firewall import count_tokens
from membrane.keyframes import keyframe, keyframes, matching_sentence, sentence_frequency, terms

FILLER = [
    line
    for i in range(30)
    for line in (
        f"We talked about the weather and the plans for day {i}.",
        "That is all for now and it was a lot.",
        "Thanks a lot. Understood. Her new one is there and that was it.",
    )
]


def _kf(text: str, cap: int = 40, limit: int = 1) -> list[str]:
    df, n = sentence_frequency([*FILLER, text])
    return keyframes(text, df, n, count_tokens, cap, limit)


def test_a_sentence_that_opens_with_a_pronoun_stays_with_the_one_before_it():
    text = "Nothing new today. The Lisbon office moved to Rua Garrett. It opens at 0730 now."
    assert _kf(text) == ["The Lisbon office moved to Rua Garrett. It opens at 0730 now."]
    # Whichever of the two is the denser one, they travel together.
    text = "We talked about the weather. Marta called from Porto. Her extension is 4417 now."
    assert _kf(text) == ["Marta called from Porto. Her extension is 4417 now."]


def test_a_pronoun_sentence_that_says_little_is_left_behind():
    text = "The Lisbon office moved to Rua Garrett 4417. That is all for now and it was a lot."
    assert _kf(text) == ["The Lisbon office moved to Rua Garrett 4417."]


def test_a_very_short_sentence_comes_along_only_when_it_is_the_densest():
    assert _kf("We talked about the plans for the day. Now 4417-Zeta.") == [
        "We talked about the plans for the day. Now 4417-Zeta."
    ]
    assert _kf("The Lisbon door code is 4417. Thanks a lot.") == ["The Lisbon door code is 4417."]
    assert _kf("The Lisbon door code is 4417. Understood. Understood.") == [
        "The Lisbon door code is 4417."
    ]


def test_up_to_three_sentences_and_never_over_the_cap():
    text = "Quarterly figures for Porto arrived. They beat the forecast. That was a first."
    assert _kf(text, cap=60) == [text]
    assert all(count_tokens(frame) <= 12 for frame in _kf(text, cap=12, limit=3))
    four = "Porto figures arrived. They beat forecast F77. That was Q3. It ends the Zeta streak."
    assert all(frame.count(".") <= 3 for frame in _kf(four, cap=80, limit=3))


def test_a_line_break_or_a_field_break_does_not_join_sentences():
    assert _kf("The Lisbon office moved to Rua Garrett.\nIt opens at 0730 now.", limit=2) == [
        "The Lisbon office moved to Rua Garrett.",
        "It opens at 0730 now.",
    ]
    serialized = "[fact] kind=note | text=Lisbon office moved to Rua Garrett. | It opens at 0730."
    assert all("Garrett. It" not in frame for frame in _kf(serialized, limit=2))


def test_several_keyframes_never_repeat_a_sentence():
    text = (
        "Gate G7001 is for Lisbon. It closes at 2200. "
        "Gate G7002 is for Porto. It closes at 2300."
    )
    frames = _kf(text, limit=2)
    assert frames == [
        "Gate G7001 is for Lisbon. It closes at 2200.",
        "Gate G7002 is for Porto. It closes at 2300.",
    ]


def test_matching_sentence_brings_the_leaning_sentence_that_holds_the_answer():
    text = "We met on Tuesday. Trip 12 went through gate G4417. It ended at locker L9920. Bye now."
    found = matching_sentence(text, terms("which locker for trip 12"), count_tokens, 40)
    assert found == "Trip 12 went through gate G4417. It ended at locker L9920."
    assert matching_sentence(text, terms("nothing shared"), count_tokens, 40) is None
    assert matching_sentence(text, terms("trip 12"), count_tokens, 12) == (
        "Trip 12 went through gate G4417."
    )


def test_single_sentence_texts_are_unchanged():
    df, n = sentence_frequency(FILLER)
    assert keyframe("The door code is 4417.", df, n, count_tokens) == "The door code is 4417."


def test_a_question_ranks_behind_every_statement():
    text = "Which Zanzibar ferry leaves from pier Q7 on Thursdays? We talked about the plans."
    assert _kf(text) == ["We talked about the plans."]
    # Among questions the densest still wins, and a text of only questions keeps one.
    assert _kf("Is that all? Which Zanzibar ferry leaves from pier Q7?") == [
        "Which Zanzibar ferry leaves from pier Q7?"
    ]
    # A statement that leans on a question still brings it along.
    text = "We talked a lot. Which ferry goes to Zanzibar? It leaves from pier Q7 at 0730."
    assert _kf(text) == ["Which ferry goes to Zanzibar? It leaves from pier Q7 at 0730."]


def test_matching_prefers_the_statement_to_the_question_that_shares_more():
    text = "What is the locker code for trip 12 again? The code for that trip is L9920. Bye now."
    wanted = terms("what is the locker code for trip 12")
    assert matching_sentence(text, wanted, count_tokens, 40) == "The code for that trip is L9920."
    only = "What is the locker code for trip 12 again?"
    assert matching_sentence(only, wanted, count_tokens, 40) == only
