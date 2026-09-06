"""Phrasing regressions for the natural-language spec parser.

The heuristic reader is pinned here with the LLM reader switched OFF (`backend="off"`):
a corpus assertion must be about the rules, never about a model's mood that day. The
LLM merge is tested separately with a stubbed reader, so the merge rule itself is pinned
without a network or a login.

Two rules are load-bearing and both have a test below:

  * an unrecognised request must stay unrecognised. Returning a default Spec dressed up
    as an interpretation is the fabrication bug the parser was rewritten to remove.
  * a reading that required a judgement call must be REPORTED, not just made. Ambiguous
    words are resolved into `assumptions`, and every interface that echoes a parse back
    to a human has to show them.
"""
from __future__ import annotations

import pytest

from eqrl.llm import spec_parser as sp
from eqrl.llm.spec_parser import parse_spec_verbose


def parse(text: str) -> sp.ParseResult:
    return parse_spec_verbose(text, backend="off")


def rec(text: str) -> dict:
    return parse(text).recognised


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
        got = rec("PCIe Gen2 CTLE, ~9 dB boost, under 12 mW")
        assert got["target_boost_db"] == pytest.approx(9.0)
        assert got["power_w_max"] == pytest.approx(0.012)

    def test_lowercase_db_and_no_space(self):
        got = rec("I want 9.8 db gain but a loss margin of 13db")
        assert got["target_boost_db"] == pytest.approx(9.8)
        assert got["channel_loss_db"] == pytest.approx(13.0)


class TestStandards:
    """Naming a link standard is how most people state a data rate."""

    @pytest.mark.parametrize("text,gbps", [
        ("PCIe Gen2 CTLE, 9 dB boost", 5.0),
        ("pcie gen3 receiver", 8.0),
        ("PCI Express Gen 4 link", 16.0),
        ("USB 3.0 front end", 5.0),
        ("SATA 3 equalizer", 6.0),
        ("10GbE SFP+ host side", 10.3125),
    ])
    def test_standard_implies_rate_and_says_so(self, text, gbps):
        r = parse(text)
        assert r.recognised["data_rate_gbps"] == pytest.approx(gbps)
        assert r.recognised["nyquist_ghz"] == pytest.approx(gbps / 2)
        assert "data_rate_gbps" in r.assumptions

    def test_explicit_rate_beats_the_standard(self):
        r = parse("PCIe Gen2 at 8 Gbps")
        assert r.recognised["data_rate_gbps"] == pytest.approx(8.0)
        assert "data_rate_gbps" not in r.assumptions

    def test_pam4_does_not_get_a_nyquist_guess(self):
        assert "nyquist_ghz" not in rec("56 Gbps PAM4 link")


class TestRanges:
    def test_boost_range_sets_bounds_and_midpoint_target(self):
        r = parse("boost between 8 and 10 dB over a 14 dB channel")
        assert r.recognised["boost_db_min"] == pytest.approx(8.0)
        assert r.recognised["boost_db_max"] == pytest.approx(10.0)
        assert r.recognised["target_boost_db"] == pytest.approx(9.0)
        assert "target_boost_db" in r.assumptions
        assert r.recognised["channel_loss_db"] == pytest.approx(14.0)

    def test_tunable_range_number_first(self):
        r = parse("3-12 dB of boost, tunable")
        assert r.recognised["boost_db_min"] == pytest.approx(3.0)
        assert r.recognised["boost_db_max"] == pytest.approx(12.0)


class TestAmbiguityIsDeclared:
    """"gain" is not a synonym for peaking -- it is a word with two referents here."""

    def test_bare_gain_is_read_as_peaking_and_says_so(self):
        r = parse("I want 9.8 db gain but a loss margin of 13db")
        assert r.recognised["target_boost_db"] == pytest.approx(9.8)
        assert "target_boost_db" in r.assumptions, (
            "reading bare 'gain' as the peaking target is a judgement call; making it "
            "without recording it is the fabrication bug in a nicer suit")
        assert "dc gain" in r.assumptions["target_boost_db"].lower()

    def test_explicit_dc_gain_is_not_an_assumption(self):
        r = parse("3 dB DC gain")
        assert r.recognised == pytest.approx({"dc_gain_db_min": 3.0})
        assert r.assumptions == {}

    def test_explicit_boost_is_not_an_assumption(self):
        r = parse("9 dB of boost")
        assert r.recognised == pytest.approx({"target_boost_db": 9.0})
        assert r.assumptions == {}


class TestWarnings:
    """Out-of-range and implausible values are reported, not clamped or planted."""

    def test_target_outside_boost_range_warns_but_keeps_the_number(self):
        r = parse("20 dB of boost")
        assert r.recognised["target_boost_db"] == pytest.approx(20.0)
        assert any("3 to 12 dB" in w for w in r.warnings)

    def test_implausible_unit_slip_is_rejected(self):
        # 15 W is a space heater, not a CTLE. A slipped SI prefix must not become a
        # budget the slider then happily displays.
        r = parse("9 dB boost, power under 15 W")
        assert "power_w_max" not in r.recognised
        assert any("power_w_max" in w for w in r.warnings)


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
    # The misspellings that used to be the rules' ceiling. Each is a real miss; the
    # fix is a spelling table, not fuzzy matching (which reads "less" as "loss").
    ("9 dB bost over a 14 dB channel", {B: 9.0, C: 14.0}),
    ("9 db boots, 14 db chanel", {B: 9.0, C: 14.0}),
    ("9 dB gian", {B: 9.0}),
    ("nine decibels of peeking, fourteen db chanel", {B: 9.0, C: 14.0}),
]


@pytest.mark.parametrize("text,want", CORPUS, ids=[t[:38] for t, _ in CORPUS])
def test_corpus(text, want):
    got = rec(text)
    for key, value in want.items():
        assert key in got, f"{key} not recognised in {text!r} (got {got})"
        assert got[key] == pytest.approx(value)


class TestNoFalsePositives:
    """The unit-less fallbacks must not claim a number that belongs to another field."""

    @pytest.mark.parametrize("text,forbidden", [
        ("CTLE for 8 Gbps NRZ", ("channel_loss_db", "target_boost_db")),
        ("PCIe Gen2 at 5 GT/s", ("channel_loss_db", "target_boost_db")),
        ("design for 2.5 GHz peak frequency", ("channel_loss_db", "target_boost_db")),
        ("I have 3 questions about the channel", ("channel_loss_db",)),
        ("eye height 100 mV and 0.4 UI", ("channel_loss_db", "target_boost_db")),
        ("boost it, less loss please", ("channel_loss_db", "target_boost_db")),
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
        r = parse(text)
        assert r.recognised == {}
        assert not r.understood

    def test_spec_is_still_usable_but_flagged_as_defaults(self):
        # `spec` always holds a runnable Spec; `understood` is how a caller tells that
        # apart from a real interpretation. The dashboard 422s on this.
        r = parse("lets fuck")
        assert r.spec is not None
        assert not r.understood


class TestLLMMerge:
    """The LLM is a second reader. Its answer is merged, never trusted over a rule."""

    @pytest.fixture
    def stub(self, monkeypatch):
        replies: dict = {}
        monkeypatch.setattr(sp, "_pick_backend", lambda backend: "cli")
        monkeypatch.setattr(sp, "_llm_cli", lambda text, model: replies["raw"])
        return replies

    def test_llm_fills_what_the_rules_missed(self, stub):
        stub["raw"] = '{"target_boost_db": 9, "power_w_max": 0.012}'
        r = parse_spec_verbose("9 dB boost and keep it frugal, 12 mW-ish")
        assert r.sources["target_boost_db"] == "both"
        assert r.recognised["power_w_max"] == pytest.approx(0.012)
        assert r.llm_backend == "cli"
        assert r.source == "heuristic+cli"

    def test_disagreement_keeps_the_rule_and_reports_it(self, stub):
        stub["raw"] = '```json\n{"channel_loss_db": 41}\n```'
        r = parse_spec_verbose("9 dB boost over a 14 dB channel")
        assert r.recognised["channel_loss_db"] == pytest.approx(14.0)
        assert r.conflicts["channel_loss_db"] == {"heuristic": 14.0, "llm": 41.0}

    def test_llm_cannot_add_unknown_or_implausible_fields(self, stub):
        stub["raw"] = '{"target_boost_db": 9000, "favourite_colour": "blue", "vdd_nominal": 1.8}'
        r = parse_spec_verbose("nothing readable here really")
        assert "favourite_colour" not in r.recognised
        assert "target_boost_db" not in r.recognised
        assert r.recognised["vdd_nominal"] == pytest.approx(1.8)
        assert any("target_boost_db" in w for w in r.warnings)

    def test_llm_failure_leaves_the_rules_standing(self, monkeypatch):
        monkeypatch.setattr(sp, "_pick_backend", lambda backend: "cli")

        def boom(text, model):
            raise RuntimeError("claude CLI exit 1: not logged in")
        monkeypatch.setattr(sp, "_llm_cli", boom)
        r = parse_spec_verbose("9 dB boost")
        assert r.recognised == pytest.approx({"target_boost_db": 9.0})
        assert r.llm_backend is None
        assert any("unavailable" in w for w in r.warnings)

    def test_llm_assumption_is_kept_only_for_its_field(self, stub):
        stub["raw"] = ('{"target_boost_db": 9, "_assumptions": {"target_boost_db": '
                       '"read gain as peaking", "channel_loss_db": "made up"}}')
        r = parse_spec_verbose("9 dB gain")
        assert "channel_loss_db" not in r.assumptions
        # the rules already declared this one; the model does not overwrite it
        assert "dc gain" in r.assumptions["target_boost_db"].lower()

    def test_empty_object_means_nothing(self, stub):
        stub["raw"] = "{}"
        r = parse_spec_verbose("hello how are you")
        assert not r.understood


class TestJsonExtraction:
    @pytest.mark.parametrize("raw,want", [
        ('{"a": 1}', {"a": 1}),
        ('Sure! ```json\n{"a": 1}\n```', {"a": 1}),
        ('{ok: true}', {"ok": True}),
        ('[1, 2]', None),
        ('no json here', None),
        ('', None),
    ])
    def test_extract(self, raw, want):
        assert sp._extract_json(raw) == want
