"""Phrasing regressions for the natural-language spec parser.

The heuristic path is what actually runs: `parse_spec_verbose` falls through to it
whenever no Anthropic credential is configured, which so far has been always. Every bug
this file pins is a real one that reached a user -- a request the parser either misread
or refused outright -- so treat a failure here as a demo failure, not a style question.

Two rules are load-bearing and both have a test below:

  * an unrecognised request must stay unrecognised. Returning a default Spec dressed up
    as an interpretation is the fabrication bug the parser was rewritten to remove.
  * a reading that required a judgement call must be REPORTED, not just made. Ambiguous
    words are resolved into `assumptions`, and every interface that echoes a parse back
    to a human has to show them.
"""
from __future__ import annotations

import pytest

from eqrl.llm.spec_parser import parse_spec_verbose


def rec(text: str) -> dict:
    return parse_spec_verbose(text).recognised


class TestOrderingTraps:
    """Both word orders, for every field that can be said either way.

    A sentence naming two dB quantities is the whole difficulty: the parser must attach
    each number to its own noun instead of to whichever came first.
    """

    def test_boost_before_channel(self):
        assert rec("8.9 dB boost over a 14 dB channel") == pytest.approx(
            {"target_boost_db": 8.9, "channel_loss_db": 14.0})

    def test_label_first_both_fields(self):
        assert rec("boost: 8.9 dB, channel loss 14 dB") == pytest.approx(
            {"target_boost_db": 8.9, "channel_loss_db": 14.0})

    def test_channel_figure_precedes_its_noun(self):
        # "13 dB of attenuation with 9.8 dB gain": a label-first rule anchored on
        # "attenuation" runs forward past 13 and claims 9.8 as the channel loss.
        got = rec("13 dB of attenuation with 9.8 dB gain")
        assert got["channel_loss_db"] == pytest.approx(13.0)
        assert got["target_boost_db"] == pytest.approx(9.8)

    def test_dc_gain_does_not_reach_across_a_clause(self):
        # "3 dB dc gain over a 14 dB channel": a free gap after the "dc gain" label runs
        # to 14 and files the CHANNEL LOSS as the DC gain.
        got = rec("9 dB boost with 3 dB dc gain over a 14 dB channel")
        assert got["dc_gain_db_min"] == pytest.approx(3.0)
        assert got["target_boost_db"] == pytest.approx(9.0)
        assert got["channel_loss_db"] == pytest.approx(14.0)


class TestVocabulary:
    """The words people actually use, as opposed to the words the schema uses."""

    @pytest.mark.parametrize("text", [
        "give me 10db peaking and 15db insertion loss",
        "10 dB of equalization against 15 dB insertion loss",
        "10dB EQ, 15dB channel",
    ])
    def test_peaking_synonyms(self, text):
        got = rec(text)
        assert got["target_boost_db"] == pytest.approx(10.0)
        assert got["channel_loss_db"] == pytest.approx(15.0)

    @pytest.mark.parametrize("text,want", [
        ("a loss margin of 13db", 13.0),
        ("13 dB of attenuation", 13.0),
        ("12 dB lossy channel", 12.0),
        ("insertion loss of 14 dB at Nyquist", 14.0),
    ])
    def test_loss_said_without_the_word_channel(self, text, want):
        assert rec(text)["channel_loss_db"] == pytest.approx(want)

    def test_bare_milliwatts_are_a_power_budget(self):
        # "under 12 mW" names no field and was silently dropped -- while being the
        # example in solve.py's own docstring.
        assert rec("PCIe Gen2 CTLE, ~9 dB boost, under 12 mW") == pytest.approx(
            {"target_boost_db": 9.0, "power_w_max": 0.012})

    def test_lowercase_db_and_no_space(self):
        got = rec("I want 9.8 db gain but a loss margin of 13db")
        assert got["target_boost_db"] == pytest.approx(9.8)
        assert got["channel_loss_db"] == pytest.approx(13.0)


class TestAmbiguityIsDeclared:
    """"gain" is not a synonym for peaking -- it is a word with two referents here."""

    def test_bare_gain_is_read_as_peaking_and_says_so(self):
        r = parse_spec_verbose("I want 9.8 db gain but a loss margin of 13db")
        assert r.recognised["target_boost_db"] == pytest.approx(9.8)
        assert "target_boost_db" in r.assumptions, (
            "reading bare 'gain' as the peaking target is a judgement call; making it "
            "without recording it is the fabrication bug in a nicer suit")
        assert "dc gain" in r.assumptions["target_boost_db"].lower()

    def test_explicit_dc_gain_is_not_an_assumption(self):
        r = parse_spec_verbose("3 dB DC gain")
        assert r.recognised == pytest.approx({"dc_gain_db_min": 3.0})
        assert r.assumptions == {}

    def test_explicit_boost_is_not_an_assumption(self):
        r = parse_spec_verbose("9 dB of boost")
        assert r.recognised == pytest.approx({"target_boost_db": 9.0})
        assert r.assumptions == {}


B, C, P = "target_boost_db", "channel_loss_db", "power_w_max"

#: A survey of how the same request gets written by people who are not reading the
#: schema. Every entry was a real miss at some point. Keep adding to it: this corpus, not
#: an impression, is what "the parser handles ordinary English" is allowed to mean.
CORPUS = [
    ("9dB boost, 14dB channel", {B: 9.0, C: 14.0}),
    ("9 DB boost over a 14 DB channel", {B: 9.0, C: 14.0}),
    ("hey can you make me a circuit with about 9 db of boost", {B: 9.0}),
    ("i need roughly 9 dB peaking please", {B: 9.0}),
    ("the channel is 14 dB, I want 9 dB of boost", {B: 9.0, C: 14.0}),
    ("over a 14 dB channel give me 9 dB peaking", {B: 9.0, C: 14.0}),
    ("The channel loses 14 dB. I need 9 dB of boost.", {B: 9.0, C: 14.0}),
    ("Boost should be 9 dB. Channel is 14 dB. Power under 10 mW.",
     {B: 9.0, C: 14.0, P: 0.010}),
    ("at least 9 dB of boost", {B: 9.0}),
    ("no more than 10 mW, 9 dB boost", {B: 9.0, P: 0.010}),
    ("9 decibels of boost", {B: 9.0}),
    ("9 dB boost, 0.01 W power budget", {B: 9.0, P: 0.010}),
    ("boost 9 dB, power 10 milliwatts", {B: 9.0, P: 0.010}),
    ("nine dB of boost", {B: 9.0}),
    # Insertion loss is quoted signed as often as not; it is a magnitude.
    ("9 dB boost over a -14 dB channel", {B: 9.0, C: 14.0}),
    # Unit omitted entirely, next to the label.
    ("9 dB gain and 13 loss", {B: 9.0, C: 13.0}),
    ("boost 9, channel 14", {B: 9.0, C: 14.0}),
    ("PCIe Gen2, 5 GT/s, equalize a 14 dB channel with 9 dB of peaking",
     {B: 9.0, C: 14.0}),
    ("CTLE for 8 Gbps NRZ, 12 dB insertion loss at Nyquist, 9 dB boost",
     {B: 9.0, C: 12.0}),
    ("I need to recover 14 dB of channel loss", {C: 14.0}),
]


@pytest.mark.parametrize("text,want", CORPUS, ids=[t[:38] for t, _ in CORPUS])
def test_corpus(text, want):
    got = rec(text)
    for key, value in want.items():
        assert key in got, f"{key} not recognised in {text!r} (got {got})"
        assert got[key] == pytest.approx(value)


#: The known ceiling. Rules cannot reach a misspelled label without fuzzy matching, and
#: fuzzy matching on a six-word vocabulary invents readings more often than it rescues
#: them. These are the cases that justify the LLM path, and they are recorded as xfail so
#: the boundary stays visible instead of being quietly dropped from the corpus.
@pytest.mark.parametrize("text", [
    "9 dB bost over a 14 dB channel",
    "9 db boots, 14 db chanel",
    "9 dB gian",
])
@pytest.mark.xfail(strict=True, reason="misspelled label; needs the LLM path")
def test_typos_are_the_ceiling(text):
    assert "target_boost_db" in rec(text)


class TestNoFalsePositives:
    """The unit-less fallbacks must not claim a number that belongs to another field."""

    @pytest.mark.parametrize("text,forbidden", [
        ("CTLE for 8 Gbps NRZ", ("channel_loss_db", "target_boost_db")),
        ("PCIe Gen2 at 5 GT/s", ("channel_loss_db", "target_boost_db")),
        ("design for 2.5 GHz peak frequency", ("channel_loss_db", "target_boost_db")),
        ("I have 3 questions about the channel", ("channel_loss_db",)),
        ("eye height 100 mV and 0.4 UI", ("channel_loss_db", "target_boost_db")),
    ])
    def test_other_units_are_not_decibels(self, text, forbidden):
        got = rec(text)
        for key in forbidden:
            assert key not in got, f"{key} wrongly read from {text!r}: {got}"


class TestRefusal:
    """An unreadable request must produce nothing. Not a default wearing a costume."""

    @pytest.mark.parametrize("text", [
        "lets fuck", "hello how are you", "", "design me something cool",
        "what is a CTLE",
    ])
    def test_nothing_recognised(self, text):
        r = parse_spec_verbose(text)
        assert r.recognised == {}
        assert not r.understood

    def test_spec_is_still_usable_but_flagged_as_defaults(self):
        # `spec` always holds a runnable Spec; `understood` is how a caller tells that
        # apart from a real interpretation. The dashboard 422s on this.
        r = parse_spec_verbose("lets fuck")
        assert r.spec is not None
        assert not r.understood
