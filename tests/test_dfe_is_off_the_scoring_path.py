"""Two structural facts about the DFE that the prose depends on, pinned so they cannot
change in silence. No simulator, no PDK — this runs in CI and it runs now.

WHY THIS FILE EXISTS
--------------------
There are two different DFEs in this repo and they share a name, which is exactly the
condition under which an honest sentence turns into a false one without anyone editing it.

  1. The one every published number was measured with is INLINE IN THE EYE ENGINE.
     `compute_eye` takes `dfe_taps` defaulting to 1, adapts the tap to the measured first
     post-cursor, and `measures.py` calls it without passing `dfe_taps` — so every reward,
     every PVT corner and every `hard_pass` verdict is scored on a POST-DFE eye.
  2. `circuits/dfe.py` and the `w_dfe` field are a separate, behavioural, hand-run
     characterisation stage with NO caller on any scoring path.

`docs/PROBLEM.md` says the first DFE "is always on" and that "every reward and all 45 PVT
corners are scored with it active", and it says `w_dfe` is a pinned, excluded field. Both
sentences are true today and both are one keyword argument away from being false. A
docstring cannot notice that. This file can.

It replaces a test that was named `test_w_dfe_is_not_a_dead_parameter` and did not touch
`w_dfe` at all — it swept a bare float tap through the behavioural stage. That check is
still worth having and still exists (renamed, in `test_monotonicity.py` and
`test_dfe.py`), but it never provided the assurance its name advertised, which is worse
than providing none: a reader budgets no further scrutiny for a claim they believe is
covered.

WHAT IT CHECKS AND WHY THAT SHAPE
----------------------------------
Both checks are over SOURCE TEXT plus one signature, because what is being pinned is a
property of how the code is WRITTEN — an argument that is absent, a field that is not
read. A behavioural test cannot see an absence; it would pass just as happily if the
scored eye quietly went pre-DFE and the docs quietly went wrong. The style is
`tests/test_citations.py`'s: read the tree, assert on what is there.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

from silq.sim.eye import compute_eye

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "silq"

MEASURES = SRC / "sim" / "measures.py"

#: Files allowed to READ `dv.w_dfe` as an attribute. Both are off every scoring path:
#: `circuits/dfe.py` is the hand-run characterisation testbench (CLI only) and
#: `schematic.py` prints the knob as a label. A read anywhere else is the regression.
ATTR_READ_ALLOWED = {"circuits/dfe.py", "schematic.py"}

#: Files allowed to MENTION `w_dfe` at all. The two beyond the set above name it only in
#: order to leave it out — a comment saying it is fixed at 0, and a dict filter dropping
#: the key — so they are evidence for the claim, not against it. `circuits/ctle.py`
#: declares the field and carries the exclusion comment.
MENTION_ALLOWED = ATTR_READ_ALLOWED | {
    "circuits/ctle.py",
    "experiments/qualify_delivered.py",
    "experiments/robust_library.py",
    # `pareto.py` names it to CARRY it, not to read it. The neighbour sweep perturbs a
    # design through `decode_action(encode_action(...))`, and the action vector is the six
    # searched parameters -- so the round trip would silently reset `w_dfe` to its default.
    # Line 77 puts the original value back. The value is never consulted: `w_dfe` appears
    # nowhere in sim/, evaluator.py, specs.py or solve.py, and measure_all on the same
    # design at w_dfe=0.0 and w_dfe=8.0 returns bit-identical measures (checked
    # 2026-09-11). It is a dict key being preserved, not a knob being used.
    "pareto.py",
}

_ATTR_READ = re.compile(r"\.\s*w_dfe\b")
_MENTION = re.compile(r"\bw_dfe\b")


def _sources() -> list[tuple[str, str]]:
    """(repo-relative-under-src/silq path, text) for every module in the package."""
    return [(p.relative_to(SRC).as_posix(), p.read_text(encoding="utf-8", errors="replace"))
            for p in sorted(SRC.rglob("*.py")) if "__pycache__" not in p.parts]


def _call_args(text: str, func: str) -> list[str]:
    """The argument text of every `func(...)` CALL in `text`, paren-balanced.

    Written by hand rather than with `ast` because an import line and a call look alike to
    a regex but not to a paren scan: `import compute_eye` is simply never followed by `(`.
    """
    out = []
    for m in re.finditer(rf"\b{func}\s*\(", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += (text[i] == "(") - (text[i] == ")")
            i += 1
        out.append(text[m.end():i - 1])
    return out


def test_compute_eye_defaults_to_a_dfe_that_is_on():
    """The scored eye's DFE is on because `compute_eye`'s own default turns it on."""
    default = inspect.signature(compute_eye).parameters["dfe_taps"].default
    assert isinstance(default, int) and default >= 1, (
        f"compute_eye's `dfe_taps` now defaults to {default!r}. Every caller that omits "
        "the argument — which is every scoring caller — silently switched to a PRE-DFE "
        "eye. docs/PROBLEM.md's 'it is always on ... every reward and all 45 PVT corners "
        "are scored with it active' becomes false the moment this default drops below 1, "
        "as does every eye number published against the 0.4 UI / 100 mV thresholds. "
        "If the change is intended, the docs and the numbers must be redone with it.")


def test_measures_does_not_override_the_dfe_default():
    """`measures.py` must keep taking that default — the other half of the same claim."""
    text = MEASURES.read_text(encoding="utf-8")
    calls = _call_args(text, "compute_eye")
    assert len(calls) >= 2, (
        f"found {len(calls)} compute_eye call sites in {MEASURES.name}; there were two "
        "(in `eye` and in `measure_all`). The scanner is probably broken, and a broken "
        "scanner makes this test pass without checking anything.")
    for args in calls:
        assert "dfe_taps" not in args, (
            f"a compute_eye call in {MEASURES.name} now passes dfe_taps explicitly:\n"
            f"    compute_eye({args.strip()})\n"
            "The scored eye no longer takes the engine default. If this passes 0, every "
            "reward, all 45 PVT corners and every hard_pass verdict are now pre-DFE "
            "numbers, and docs/PROBLEM.md's 'the eye metric in `measure_all` is post-DFE "
            "... every reward and all 45 PVT corners are scored with it active' is false. "
            "Fix the docs and re-measure, or revert the argument.")


def test_w_dfe_is_read_on_no_scoring_path():
    """`dv.w_dfe` is read only where it cannot reach a published number."""
    offenders = {rel: sorted({i for i, ln in enumerate(text.splitlines(), 1)
                              if _ATTR_READ.search(ln)})
                 for rel, text in _sources() if _ATTR_READ.search(text)}
    assert set(offenders) == ATTR_READ_ALLOWED, (
        f"`dv.w_dfe` is read in {sorted(offenders)}; the only files allowed to read it "
        f"are {sorted(ATTR_READ_ALLOWED)} (lines: {offenders}). A read outside those "
        "makes the exclusion comment where `w_dfe` is left out of the action vector "
        "(`ctle.py:97-99`) false: the field is pinned at 0.0 and never chosen by the "
        "agent, so a scoring path that reads it is scoring a knob nothing optimises. If "
        "the intent is to make the tap live, it has to become an action variable, and "
        "the architecture freeze forbids that without retraining.")


def test_w_dfe_is_not_even_mentioned_off_the_allow_list():
    """The stricter net: a read spelled `getattr(dv, "w_dfe")` or `d["w_dfe"]` carries no
    dot and would slip past the check above, so pin every textual mention too."""
    mentions = {rel for rel, text in _sources() if _MENTION.search(text)}
    assert len(mentions) >= 3, (
        f"only {sorted(mentions)} mention w_dfe; the field is declared in "
        "circuits/ctle.py and used in circuits/dfe.py and schematic.py, so the source "
        "scan has probably stopped finding files.")
    assert mentions <= MENTION_ALLOWED, (
        f"w_dfe is now named in {sorted(mentions - MENTION_ALLOWED)}. Every existing "
        "mention is either the declaration, the characterisation testbench, a drawing "
        "label, or an explicit exclusion; a new one is most likely a new read on a path "
        "that feeds a published number. See the reasoning in "
        "test_w_dfe_is_read_on_no_scoring_path, and add the file here only after "
        "confirming the mention cannot influence a measured result.")
