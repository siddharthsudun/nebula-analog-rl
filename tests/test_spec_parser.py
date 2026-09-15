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

from silq.llm import spec_parser as sp
from silq.llm.spec_parser import parse_spec_verbose


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


class TestPeakBand:
    """The requested peak band, which is the one field that REORDERS the search.

    Every other requirement only judges a candidate at the end. A peak band is different:
    `pipeline._search_band` turns it into the corpus-first start order, and
    `g32_solve`'s AIM is the band's geometric centre. So a band this reader invents is
    not a cosmetic misread -- it sends the search somewhere the user never asked for,
    and a band it MISSES leaves the search steering on the competition default while the
    verdict is judged against the ask.
    """

    @pytest.mark.parametrize("text", [
        "peak between 2 and 2.5 GHz",
        "peak from 2 to 2.5 GHz",
        "peaking band 2.0-2.5 GHz",
        "peak band 2 GHz to 2.5 GHz",
        "peak in the 2 - 2.5 GHz band",
        "9 dB boost peaking between 2.0 and 2.5 GHz",
        "peak band 2000 to 2500 MHz",
    ])
    def test_a_stated_band_is_read_verbatim(self, text):
        got = rec(text)
        assert got["peak_freq_lo_ghz"] == pytest.approx(2.0)
        assert got["peak_freq_hi_ghz"] == pytest.approx(2.5)
        assert "peak_freq_lo_ghz" not in parse(text).assumptions, (
            "a band the user bounded themselves required no judgement call")

    def test_an_inverted_band_is_sorted_not_passed_through(self):
        """`hard_pass`'s peak_in_band is `lo <= f <= hi`. Inverted, nothing satisfies it.

        Sorting is not tidiness: an unsorted pair reaches the acceptance spec as a check
        that fails every design, and the run reports "no design met the requirement"
        about a requirement the user never actually set.
        """
        got = rec("peak between 2.5 and 2 GHz")
        assert (got["peak_freq_lo_ghz"], got["peak_freq_hi_ghz"]) == pytest.approx((2.0, 2.5))

    @pytest.mark.parametrize("text,centre", [
        ("peak the response at 3 GHz", 3.0),
        ("peak frequency of 2.4 GHz", 2.4),
        ("peaking frequency 2.4 GHz", 2.4),
        ("put the peak at 4 GHz", 4.0),
        ("peak near 2.4 GHz", 2.4),
        ("centre the peak on 3.5 GHz", 3.5),
        ("peak gain around 1.8 GHz", 1.8),
        ("fpeak 2.2 GHz", 2.2),
        ("peak at 2400 MHz", 2.4),
        ("peaking centered at 3 GHz", 3.0),
        ("a 3 GHz peak, 9 dB of boost", 3.0),
    ])
    def test_a_point_ask_becomes_a_declared_band(self, text, centre):
        """A point names a centre; the Spec stores two edges. The width is a judgement
        call, so it is made once, in one constant, and REPORTED -- the same contract the
        bare-"gain" reading is held to."""
        r = parse(text)
        tol = sp.PEAK_POINT_TOL
        assert r.recognised["peak_freq_lo_ghz"] == pytest.approx(centre * (1 - tol))
        assert r.recognised["peak_freq_hi_ghz"] == pytest.approx(centre * (1 + tol))
        for key in ("peak_freq_lo_ghz", "peak_freq_hi_ghz"):
            assert key in r.assumptions, f"{key} was widened without telling the user"

    @pytest.mark.parametrize("text", [
        "peaking between 8 and 10 dB",              # a boost range, not a band
        "peak gain of 9 dB on an 8 Gbps link",      # the rate is not the peak
        "9 dB of peaking for a 5 GHz Nyquist link",  # nor is Nyquist, said before it
        "9 dB peaking at 4 GHz Nyquist",            # nor said after it
        "peaking EQ for an 8 Gbps link",
        "peaking CTLE, supply 1.2 V, 0.02 mm2",
        "peaking EQ with 4 GHz of bandwidth",       # nor the bandwidth
        "Nyquist 4 GHz",
    ])
    def test_another_quantity_in_GHz_is_never_the_peak_band(self, text):
        got = rec(text)
        assert "peak_freq_lo_ghz" not in got and "peak_freq_hi_ghz" not in got, (
            f"invented a peak band from {text!r}: {got}")


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


class TestTheKeySurfaceCannotDrift:
    """The four places a field name has to appear, checked against each other.

    A field can be added to `Spec` and wired through `REQUIREMENT_FIELDS` while the LLM
    never learns it exists, or advertised to the LLM and then dropped by the merge, or
    accepted with no plausibility box so a unit slip reaches a slider. Each of those is a
    one-line omission in a different file and none of them fails anything today -- the
    request simply comes back with the field missing, which is indistinguishable from the
    user not having asked for it.
    """

    @staticmethod
    def _advertised() -> set:
        import re
        m = re.search(r"Allowed keys:(.*?)\.\n", sp._SYSTEM, re.S)
        assert m, "the system prompt no longer lists its allowed keys"
        return set(re.findall(r"[a-z0-9_]+", m.group(1)))

    def test_nothing_advertised_is_dropped_by_the_merge(self):
        assert self._advertised() - sp.SPEC_FIELDS == set()

    def test_nothing_mergeable_is_hidden_from_the_llm(self):
        assert sp.SPEC_FIELDS - self._advertised() == set()

    def test_every_advertised_key_has_a_plausibility_box(self):
        assert self._advertised() - set(sp.PLAUSIBLE) == set()

    def test_every_settable_requirement_is_something_the_llm_can_return(self):
        from silq.pipeline import REQUIREMENT_FIELDS
        assert set(REQUIREMENT_FIELDS) - self._advertised() == set()

    def test_the_llm_is_told_the_same_point_ask_convention_the_rules_use(self):
        """Both readers must turn "peak at 3 GHz" into the same band, or every point ask
        arrives as a conflict the user has to arbitrate over a convention neither of them
        chose."""
        assert "peak_freq_lo_ghz to 0.9x" in sp._SYSTEM
        assert sp.PEAK_POINT_TOL == 0.10, (
            "the prompt spells 0.9x/1.1x literally; a different tolerance here would make "
            "the two readers disagree on every point ask")


class TestOneSidedPeakAsk:
    """A comparator bounds ONE edge. The point rule must not swallow these.

    Before this existed, "peak below 3 GHz" went to the point rule and came back as
    2.7-3.3 GHz: a 2.7 GHz FLOOR the user never stated, and a ceiling ABOVE the one they
    did. The run was then scored against a band that contradicts the request on both
    sides, and `hard_pass`'s peak_in_band reported it satisfied.
    """

    @pytest.mark.parametrize("text,lo,hi", [
        ("peak below 3 GHz", None, 3.0),
        ("keep the peak under 2.2 GHz", None, 2.2),
        ("peaking up to 2.4 GHz", None, 2.4),
        ("peak not above 2.2 GHz", None, 2.2),
        ("peak at most 2.0 GHz", None, 2.0),
        ("peak less than 1.9 GHz", None, 1.9),
        ("peak above 2 GHz", 2.0, None),
        ("peak at least 1.8 GHz", 1.8, None),
        ("peak no lower than 1.6 GHz", 1.6, None),
        ("peak greater than 1.7 GHz", 1.7, None),
        ("peaking over 1.5 GHz", 1.5, None),
        ("peak below 2400 MHz", None, 2.4),
    ])
    def test_only_the_named_edge_moves(self, text, lo, hi):
        spec = parse(text).spec
        d = sp.Spec()      # the competition band, untouched
        if lo is None:
            assert spec.peak_freq_lo_ghz == pytest.approx(d.peak_freq_lo_ghz), (
                "the unbounded edge must stay at the competition default: the user asked "
                "nothing about it, and answering anyway is the fabrication this parser "
                "exists to prevent")
            assert spec.peak_freq_hi_ghz == pytest.approx(hi)
        else:
            assert spec.peak_freq_lo_ghz == pytest.approx(lo)
            assert spec.peak_freq_hi_ghz == pytest.approx(d.peak_freq_hi_ghz)

    @pytest.mark.parametrize("text,field", [
        ("peak below 3 GHz", "peak_freq_hi_ghz"),
        ("peak at least 1.8 GHz", "peak_freq_lo_ghz"),
    ])
    def test_the_one_sided_reading_is_declared(self, text, field):
        """It is still a judgement call, so it still has to be visible."""
        res = parse(text)
        assert field in res.assumptions
        assert "default" in res.assumptions[field]

    def test_a_two_sided_ask_still_wins_over_the_comparator_rule(self):
        """"peak from 1.5 to 2.0" contains "to", which is also a one-sided word."""
        spec = parse("peak from 1.5 to 2.0 GHz").spec
        assert (spec.peak_freq_lo_ghz, spec.peak_freq_hi_ghz) == pytest.approx((1.5, 2.0))

    def test_the_prompt_teaches_the_same_convention(self):
        """The two readers must agree, or every one-sided ask arrives as a conflict."""
        for word in ("one-sided", "peak_freq_hi_ghz for below", "leave the other edge"):
            assert word in sp._SYSTEM, f"the LLM prompt does not mention {word!r}"


@pytest.mark.parametrize("text", [
    "8 dB peaking, 5 GHz baud rate",
    "6 dB peaking, 10 GHz symbol rate",
    "9 dB peaking, 5 GHz data rate",
    "7 dB peaking at 4 GHz baud",
])
def test_a_rate_word_AFTER_the_figure_is_not_a_peak_request(text):
    """`_GAP_NOFREQ` only guards words BEFORE the number.

    "8 dB peaking, 5 GHz baud rate" names the link. Read as a peak request it plants a
    4.5-5.5 GHz band, which is outside the tunable range entirely -- so the search is
    steered at something the circuit cannot do, for a requirement nobody made.
    """
    spec = parse(text).spec
    d = sp.Spec()
    assert (spec.peak_freq_lo_ghz, spec.peak_freq_hi_ghz) == pytest.approx(
        (d.peak_freq_lo_ghz, d.peak_freq_hi_ghz))
