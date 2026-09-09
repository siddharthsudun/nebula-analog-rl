"""An artifact path in prose is a promise that a reader can open the file.

WHY THIS EXISTS
---------------
`tests/test_citations.py` checks that a `somefile.py:<line>` pointer resolves to code that
supports the sentence citing it. This file checks the other half of the same promise: that
a `results/foo.json` named in a document is something a reader can actually get to.

It was written after three separate discoveries on 07 Sep 2026, each found by hand and none
by CI:

1. `docs/RESULTS_INFERENCE_MODES_V2.md` sourced the `auto` routing rationale to
   `results/mode_sweep_seed99.json`, which appears in **no commit in this repository**. The
   file exists on the author's disk. Nobody else can check the number it backs.
2. `docs/PREREG_G32_ACCEPTANCE.md` justified a load-bearing design rule with a statistic
   that turned out to be assembled from misread fragments. Refuting it required
   `results/thinking_adaptive_before_after.json` — also uncommitted, so the *retraction*
   had the same defect as the claim.
3. `RESULTS.md` said of `results/legacy_design_recheck.json`: "It is kept in the tree as a
   record". The file is neither committed nor on disk. That sentence is not a stale
   pointer, it is a false statement of fact about this repository, in the document a judge
   reads first.

The three cases are different failures and get three different tests. The third is the
worst and is the only one with zero tolerance: being uncommitted is a debt a reader can be
told about, but *claiming* something is in the tree when it is not is untrue regardless of
what else the paragraph says.

WHAT IS AND IS NOT A DEFECT
---------------------------
Not every cited artifact can or should be committed. `results/surrogate_corpus.npz` is
18 MB and regenerable; `.gitignore` excludes it on purpose and says why. The rule this file
enforces is therefore not "commit everything" but:

    a cited artifact must be TRACKED, or DELIBERATELY IGNORED (a `.gitignore` rule covers
    it, so its absence is a decision rather than an oversight), or NAMED AS REGENERABLE by
    the command that makes it.

An artifact that is none of those three is a dead reference: not in the tree, not
excluded on purpose, and with no stated way to obtain it.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Artifact paths look like `results/foo.json`. Restricted to the data directories and to
#: data extensions: source files are `test_citations.py`'s job, not this one.
_ARTIFACT = re.compile(
    r"\b((?:results|scratchpad|data|artifacts)/[A-Za-z0-9_./-]+"
    r"\.(?:json|npz|zip|csv|jsonl|pt|pkl))")

#: A field citation is deliberately machine-readable: ``artifact.json → nested.field``.
#: A path alone is an artifact citation, covered by the tests below. Adding a field name
#: turns it into a claim about that artifact's schema and must be checked against the
#: on-disk JSON rather than trusted as prose.
_JSON_FIELD_CITATION = re.compile(
    r"`(?P<artifact>(?:results|scratchpad|data|artifacts)/[A-Za-z0-9_./-]+\.json)"
    r"\s*(?:→|->)\s*"
    r"(?P<field>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*`")

#: A sentence asserting the artifact is present in the repository. These are the claims
#: that must be literally true, not merely well-intentioned.
_IN_TREE = re.compile(
    r"(kept in the tree|in the tree|committed to (?:the|this) repo|checked in|"
    r"is in the repo|shipped in the repo)", re.I)

#: The honest counterpart: prose that names a missing artifact AND says it is missing. This
#: is not a defect, it is the fix -- a number whose backing run is gone should still say
#: which run it was, so the debt is nameable and re-runnable. Without this, the test would
#: push authors to DELETE the provenance rather than disclose it, which is the opposite of
#: what it is for: it would score "10 of 45 corners fail" with no source at all as clean,
#: and the same sentence naming its lost artifact as dirty.
#: Matched against a WHITESPACE-NORMALIZED window (see `_flat`), because these disclosures
#: routinely wrap across lines and markdown bold markers land mid-phrase.
_ABSENT_DECL = re.compile(
    r"(?:not|no longer|never|neither)\b[^.]{0,60}?"
    r"(?:in the tree|in tree|committed|on disk|present|available|exists?)"
    r"|is gone\b|are gone\b"
    # An ADVERB BETWEEN THE AUXILIARY AND THE VERB defeated the two literal alternatives
    # this replaces (`has been deleted`, `were deleted`). HANDOVER.md discloses the two
    # `*_design_recheck.json` artifacts as "were **deliberately deleted** in that commit
    # because they recorded the defect" -- exactly the disclosure this pattern exists to
    # honour -- and it missed it on the word "deliberately", failing the suite on correct
    # prose. Widened to `deleted` only, never `removed`: "removed" is common enough in
    # unrelated prose that admitting it could let a genuine false presence-claim past the
    # zero-tolerance sibling test, which shares this pattern. Fixed by silq-main, 07 Sep.
    r"|(?:is|are|was|were|has|have|had|been)\b[^.]{0,40}?\bdeleted\b", re.I)


def _flat(s: str) -> str:
    """Collapse whitespace and drop markdown emphasis so phrase patterns survive wrapping."""
    return re.sub(r"\s+", " ", s.replace("**", "").replace("*", ""))

#: Artifacts that are not in the tree by design, each mapped to the command that produces
#: it. Adding a row here is a claim that the command works; it is the price of citing a
#: file a reader cannot open, and it keeps the exemption from being a silent shrug.
REGENERABLE = {
    "results/seq_agent.zip":
        "python -m eqrl.agents.train_sequential --out results/seq_agent.zip",
    "results/surrogate_corpus.npz":
        "python -m eqrl.experiments.build_surrogate_corpus  (~20 min from results/raw)",
    "results/feasibility_band.json":
        "python -m eqrl.experiments.feasibility_band",
    # Prospective: docs/PROTOCOL_MODE_ERROR_AUDIT_20260909.md specifies this output path
    # for a run that has NOT been performed and is not authorized here. What was verified
    # is the plan path -- the module runs without --execute and prints its budgets without
    # starting a simulator. The confirmatory run's own ceiling is 864 searches / 166,752
    # measure_all calls / 144 h of child time, so this row states how to obtain the file,
    # not that obtaining it is cheap. Do not silently shrink the sample to make it so.
    "results/mode_error_audit_20260909/observations.json":
        "PYTHONPATH=src python -m eqrl.experiments.mode_error_audit --execute "
        "--out-dir results/mode_error_audit_20260909  (plan-only without --execute)",
}

#: Ratchet. Cited artifacts that are on disk but untracked AND not covered by a .gitignore
#: rule -- i.e. nobody decided they should be absent, they just are. Measured at 07 Sep
#: 2026 as 8. This number may go DOWN. Raising it means choosing to publish another number
#: a reader cannot check, and should be a deliberate, argued act.
#:
#: Lowered to 1 on 09 Sep 2026 by committing the seven artifacts rather than by relaxing
#: anything: the pvt_audit, target_audit, hedge_band and mode_sweep results had been sitting
#: untracked while documents quoted their numbers. The one that remains is a scratchpad
#: file, which is the category this repo deliberately does not commit.
MAX_UNCHECKABLE = 1


def _docs() -> dict[str, str]:
    out = {}
    for p in sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").glob("*.md")):
        out[p.relative_to(ROOT).as_posix()] = p.read_text(encoding="utf-8",
                                                          errors="replace")
    return out


def test_documented_json_fields_exist_in_the_named_artifact():
    """Every ``artifact.json → field.path`` citation must resolve in its JSON object.

    The check is generic because stale field attribution is not confined to PVT reports:
    it applies to every root-level Markdown document and every file in ``docs/``. The
    arrow form is intentional. It keeps an ordinary artifact citation readable while
    giving this test an unambiguous artifact-to-field contract to validate.
    """
    decoded: dict[str, object] = {}
    bad = []
    for rel, text in _docs().items():
        for match in _JSON_FIELD_CITATION.finditer(text):
            artifact, field = match.group("artifact"), match.group("field")
            path = ROOT / artifact
            if artifact not in decoded:
                try:
                    decoded[artifact] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    bad.append(f"{rel}: `{artifact}` for `{field}` is unreadable: {exc}")
                    continue

            value = decoded[artifact]
            missing = None
            for key in field.split("."):
                if not isinstance(value, dict) or key not in value:
                    missing = key
                    break
                value = value[key]
            if missing is not None:
                bad.append(f"{rel}: `{artifact} → {field}` has no `{missing}` key")

    assert not bad, (
        "documentation cites JSON fields that their named artifacts do not contain:\n  "
        + "\n  ".join(bad)
        + "\n\nCorrect the artifact, the field path, or the documentation. Do not remove "
          "the arrow citation: it is the machine-checkable link between the claim and "
          "its evidence.")


def _tracked() -> set[str]:
    r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("not a git checkout; tracked-ness is undecidable here")
    return {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}


def _is_ignored(rel: str) -> bool:
    """True if a .gitignore rule covers this path, i.e. its absence is a decision.

    `git check-ignore` answers for paths that do not exist, which is exactly the case
    that matters here -- an artifact both absent and ignored is deliberately absent.
    """
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", rel],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0


#: `--out results/foo.json` in a reproduction command is a path the reader will CREATE, not
#: a claim that it already exists. Counting those as dead references would make this test
#: fire on `docs/REPRODUCE.md` for doing its job correctly -- and a test that cries wolf on
#: correct prose gets suppressed, which costs more than the defect it was meant to catch.
_OUTPUT_ARG = re.compile(
    r"(?:--out|--output|--out-?file|-o)[= ]\s*$")


def _cited() -> dict[str, set[str]]:
    """artifact path -> set of docs citing it AS PRESENT.

    Excludes two forms that are not presence claims: a path given as a command's output
    argument (the reader creates it), and a path whose surrounding prose explicitly says it
    is missing (the author disclosed it).
    """
    cites: dict[str, set[str]] = {}
    for rel, txt in _docs().items():
        for m in _ARTIFACT.finditer(txt):
            if _OUTPUT_ARG.search(txt[max(0, m.start() - 24):m.start()]):
                continue
            # Window spans the sentence break after the path: the `.json` extension's own
            # period ends the sentence, so the disclosure usually lands just after it.
            window = _flat(txt[max(0, m.start() - 320):m.end() + 320])
            if _ABSENT_DECL.search(window):
                continue
            cites.setdefault(m.group(1), set()).add(rel)
    return cites


#: A sentence that opens with one of these refers back to the previous sentence's subject.
#: Needed because the defect this test was written for spans a sentence break:
#:
#:     ... fails 10 of 45 corners; `results/legacy_design_recheck.json`. It is kept in
#:     the tree as a record, not as the delivered result.
#:
#: The artifact ends its sentence -- the `.json` extension's own period IS the boundary --
#: so a per-sentence scan finds "kept in the tree" in a sentence containing no path at all.
#: The first draft of this test passed against the very text it was written to catch, and
#: that was only noticed because the failure was expected and did not arrive. Inheritance
#: is restricted to an opening pronoun rather than to any preceding sentence, so the claim
#: is still attributed to something it demonstrably refers to.
_ANAPHOR = re.compile(r"^\s*(?:It|This|That|These|Those|Both|They)\b")


def _sentences(txt: str) -> list[str]:
    return re.split(r"(?<=[.)])\s+", txt)


def test_no_document_claims_an_absent_artifact_is_in_the_tree():
    """ZERO TOLERANCE. "It is kept in the tree" must be true of the tree.

    This is the `legacy_design_recheck.json` defect. A stale *pointer* misleads a reader
    about where something is; a false *presence* claim tells them the repository contains
    evidence it does not contain. In a competition submission that is the difference
    between an error and an overstatement, so this test has no ratchet and no allow-list.
    """
    tracked = _tracked()
    bad = []
    for rel, txt in _docs().items():
        sents = _sentences(txt)
        for i, sentence in enumerate(sents):
            if not _IN_TREE.search(sentence):
                continue
            if _ABSENT_DECL.search(_flat(sentence)):
                continue                  # "no longer in the tree" is a disclosure, not a claim
            arts = _ARTIFACT.findall(sentence)
            # A claim that opens with a pronoun is about the previous sentence's artifact.
            if not arts and i and _ANAPHOR.match(sentence):
                arts = _ARTIFACT.findall(sents[i - 1])
            for art in arts:
                if art not in tracked:
                    on_disk = (ROOT / art).is_file()
                    bad.append(f"{rel}: claims `{art}` is in the tree; it is NOT tracked"
                               f" ({'present on disk only' if on_disk else 'ABSENT entirely'})"
                               f" -- sentence: {sentence.strip()[:160]}")
    assert not bad, (
        "a document states that an artifact is in this repository when it is not:\n  "
        + "\n  ".join(bad)
        + "\n\nFix the SENTENCE or commit the FILE. Do not add an exemption: this test "
          "exists because 'kept in the tree as a record' was written about a file that "
          "was in no commit and on no disk.")


def test_cited_artifacts_are_tracked_ignored_or_declared_regenerable():
    """A cited artifact a reader can neither open nor obtain is a dead reference."""
    tracked = _tracked()
    dead = []
    for art, docs in sorted(_cited().items()):
        if art in tracked or art in REGENERABLE:
            continue
        if (ROOT / art).is_file():
            continue                      # obtainable by the author; scored by the ratchet
        if _is_ignored(art):
            continue                      # deliberately excluded, and says so in .gitignore
        dead.append(f"`{art}` (cited by {', '.join(sorted(docs))})")
    assert not dead, (
        "these artifacts are cited in prose but are not committed, not on disk, not "
        "covered by a .gitignore rule, and not listed in REGENERABLE:\n  "
        + "\n  ".join(dead)
        + "\n\nEither commit it, add a .gitignore rule (which records that its absence is "
          "deliberate), or add it to REGENERABLE with the command that builds it. Citing "
          "a file nobody can obtain is a pointer to nothing.")


def test_uncheckable_artifact_citations_do_not_grow():
    """Ratchet on the softer failure: present for the author, absent for everyone else.

    These are not dead references -- the file exists -- but every number sourced to one is
    a number a reader must take on trust. That is a real cost, so it is counted and capped
    rather than tolerated silently.
    """
    tracked = _tracked()
    uncheckable = {}
    for art, docs in sorted(_cited().items()):
        if art in tracked or art in REGENERABLE:
            continue
        if not (ROOT / art).is_file():
            continue                      # dead references are the previous test's job
        if _is_ignored(art):
            continue                      # excluded on purpose, with a stated reason
        uncheckable[art] = sorted(docs)

    assert len(uncheckable) <= MAX_UNCHECKABLE, (
        f"{len(uncheckable)} cited artifacts exist only on the author's disk, above the "
        f"ratchet of {MAX_UNCHECKABLE}:\n  "
        + "\n  ".join(f"`{a}` <- {', '.join(d)}" for a, d in uncheckable.items())
        + "\n\nCommit it, or .gitignore it with a reason, or stop citing it. Raising "
          "MAX_UNCHECKABLE publishes another unverifiable number.")


def test_the_frozen_baseline_is_actually_present():
    """The one artifact whose absence would make the headline unreproducible.

    Every published benchmark number derives from `results/seq_clean40k.zip`. `.gitignore`
    un-ignores it explicitly (against a blanket `results/*.zip`) for exactly this reason,
    so this test pins that the exception still works -- a reordered ignore rule would
    silently drop the frozen policy from the tree and nothing else would notice.
    """
    rel = "results/seq_clean40k.zip"
    assert rel in _tracked(), (
        f"{rel} is no longer tracked. It is the frozen policy behind every published "
        "number; `.gitignore`'s `!results/seq_clean40k.zip` exception has stopped working "
        "or the file was removed. Reproduction is impossible without it.")
    assert (ROOT / rel).is_file(), f"{rel} is tracked but missing from the working tree."
