"""Every `file.py:NNN` pointer in the prose must land on what the prose says is there.

WHY THIS TEST EXISTS, AND WHY IT IS NOT PEDANTRY
------------------------------------------------
This repo argues for its results in prose, and the prose cites code by line number. On
07 Sep 2026 an audit of all 32 such citations found **13 stale** — 41%. Not one had ever
failed anything, because nothing checked them. Samples of what they pointed at:

    README.md            -> ctle.py line 29       the area-supremum claim; landed on an
                                                  unrelated comment about i_tail
    PASS_VS_VALID.md     -> pass_vs_valid line 54  landed on a BLANK LINE
    PROBLEM.md           -> ctle.py line 96        ACTION_SPACE; landed on a bare "#"
    PREREG_G32_...md     -> pipeline lines 497-525 _reselect_for_requirements, the
                                                  preregistered NULL; off by ~50 lines

Most had rotted the same day, from edits that added comments *above* the cited symbol.
That is the whole problem: a line number is a claim about a location, and it is falsified
by edits to entirely unrelated text. Prose review cannot catch it — the sentence still
reads true — and a reader who follows the pointer lands somewhere irrelevant and now
distrusts a statement that was correct.

The rule this project already applies to numbers ("a number in prose is an assertion with
no test behind it") applies to pointers, and this file is that test.

WHAT IT CHECKS, AND THE ONE HEURISTIC IT LEANS ON
--------------------------------------------------
For each citation it takes the backtick-quoted identifiers that BELONG TO IT — the text
from the end of the previous citation up to this one, looking back at most two lines —
and requires one to appear inside the cited span. The attribution matters: a first draft
of this test took identifiers from a flat 3-line window and failed six correct citations,
because a line can carry two pointers and the second inherited the first's symbols.

Matching is case-insensitive and also tries an identifier's last dotted segment, so
`Spec.dc_gain_db_min` matches the field's declaration and prose `solved` matches `SOLVED`.

WHAT THIS CANNOT CATCH, STATED PLAINLY. A citation whose sentence names no symbol and
lands on a wrong-but-non-blank line passes. That is not hypothetical: the README's
area-supremum pointer was exactly that shape, and this test would have missed it. So a
second test ratchets the number of unattributable citations — it may fall, never rise.
The fix for one is to name a symbol in the sentence, which makes it checkable forever.

Line numbers are load-bearing here, so this test's own examples above are given as file
names WITHOUT line numbers on purpose.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Where prose lives. `results/` is excluded on purpose: it holds ~2.36M generated files
#: and nothing in it is hand-written prose making a citation.
SEARCH_ROOTS = ("README.md", "RESULTS.md", "SETUP.md", "HANDOVER.md", "docs", "src",
                "tests", "scratchpad", "web/index.html", "site/index.html", "server.py")
SEARCH_EXTS = ("*.md", "*.py", "*.html", "*.js")

_CITE = re.compile(r"([A-Za-z0-9_./\\-]+\.(?:py|md|html|js)):(\d+)(?:\s*-\s*(\d+))?")
_IDENT = re.compile(r"`([^`\n]+)`")
#: Words that appear inside backticks but say nothing about a location.
_NOISE = {"None", "True", "False", "ok", "n", "k", "x", "m", "spec", "self", "dv"}
#: An identifier is a code token, not a phrase. Backticks alternate along a line, so the
#: PROSE BETWEEN two citations is captured by _IDENT too -- ` now reads "No floor-budget
#: escalation", and ` was extracted as a symbol and duly failed to appear in any code.
#: Requiring no whitespace is what separates a name from the sentence around it.
_LOOKS_LIKE_CODE = re.compile(r"""^[A-Za-z_][A-Za-z0-9_.\[\]()'"-]*$""")
#: `pipeline.py:219 now reads "No floor-budget escalation"` quotes the target's own text.
#: That is a STRONGER claim than naming a symbol -- it asserts the words at that line --
#: so it is checked the same way, and a citation carrying one is not "unattributable".
#:
#: THE CUE VERB IS REQUIRED. A bare quoted string is not a quotation of the target: in a
#: .py file it is usually just a string literal, and `"unit": "... final_report.py:272"`
#: was duly read as a claim about final_report's text. "reads"/"says"/"states" is what
#: turns a quotation mark into an assertion about what is at that line.
_QUOTED = re.compile(r'(?:reads?|says?|states?)\s+["“]([^"”\n]{6,})["”]')
_DEF = re.compile(r"^(\s*)(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")


def _iter_prose_files():
    for r in SEARCH_ROOTS:
        p = ROOT / r
        if p.is_file():
            yield p
        elif p.is_dir():
            for ext in SEARCH_EXTS:
                for q in p.rglob(ext):
                    if "__pycache__" not in q.parts:
                        yield q


def _resolve(target: str) -> Path | None:
    """Citations are written repo-relative, src-relative, or bare. Accept all three,
    but ONLY when the bare name is unambiguous -- a guess is what this test is against."""
    t = target.replace("\\", "/")
    for cand in (ROOT / t, ROOT / "src" / t):
        if cand.is_file():
            return cand
    base = t.split("/")[-1]
    hits = [q for d in ("src", "tests", "scratchpad", "docs")
            for q in (ROOT / d).rglob(base) if "__pycache__" not in q.parts]
    if (ROOT / base).is_file():
        hits.append(ROOT / base)
    return hits[0] if len(hits) == 1 else None


def _enclosing_scopes(lines: list[str], lineno: int) -> list[str]:
    """def/class names enclosing `lineno`, outermost first.

    A citation may legitimately point at a STATEMENT rather than a declaration -- "the eye
    metric in `measure_all` is post-DFE (measures.py:174)" means line 174 sits inside
    `measure_all`, and that is the normal way to cite a step of an algorithm. Without this
    the test would push every citation onto `def` lines, which is worse prose.
    """
    scopes, indents = [], []
    for i in range(min(lineno, len(lines))):
        m = _DEF.match(lines[i])
        if not m:
            continue
        col = len(m.group(1))
        while indents and indents[-1] >= col:
            indents.pop(), scopes.pop()
        indents.append(col), scopes.append(m.group(2))
    return scopes


def _citations():
    """(citing file, citing line no, citing text, target path, lo, hi) for every pointer."""
    out = []
    for f in _iter_prose_files():
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:                                   # pragma: no cover - unreadable
            continue
        for i, ln in enumerate(lines, 1):
            for m in _CITE.finditer(ln):
                lo = int(m.group(2))
                hi = int(m.group(3)) if m.group(3) else lo
                # Attribute identifiers to THIS citation: the text running back to the
                # end of the previous citation, across at most two wrapped lines. A flat
                # window would hand this pointer the previous pointer's symbols.
                # `i` is 1-based, so the two PRECEDING lines are [i-3:i-1]. Slicing to
                # [:i] instead includes the current line, and every citation then finds
                # ITSELF as its own predecessor and gets an empty context -- which fails
                # open, to the weak non-blank check. Silent under-checking, so: tested by
                # `test_attribution_window_excludes_the_citation_itself` below.
                window = "\n".join(lines[max(0, i - 3):i - 1]) + "\n" + ln[:m.start()]
                prev = None
                for pm in _CITE.finditer(window):
                    prev = pm
                ctx = window[prev.end():] if prev else window
                # A "now reads X" cue usually trails its pointer, so quoted claims are
                # looked for on BOTH sides -- up to the next citation, which owns the rest.
                after = ln[m.end():] + "\n" + "\n".join(lines[i:i + 1])
                nxt = _CITE.search(after)
                out.append((f, i, ctx, after[:nxt.start()] if nxt else after,
                            m.group(1), lo, hi))
    return out


CITATIONS = _citations()


def test_citation_corpus_is_not_empty():
    """A resolver bug that found nothing would make every test below pass vacuously."""
    assert len(CITATIONS) >= 20, (
        f"only {len(CITATIONS)} file:line citations found; this repo had 32 on "
        "07 Sep 2026. The scanner is probably broken, and a broken scanner makes the "
        "rest of this file pass without checking anything.")


@pytest.mark.parametrize("cite", CITATIONS,
                         ids=[f"{c[0].name}:{c[1]}->{c[4].split('/')[-1]}:{c[5]}"
                              for c in CITATIONS])
def test_citation_lands_on_what_it_claims(cite):
    src, srcline, ctx, after, target, lo, hi = cite
    where = f"{src.relative_to(ROOT)}:{srcline}"

    tgt = _resolve(target)
    assert tgt is not None, (
        f"{where} cites `{target}:{lo}` and that path does not resolve to exactly one "
        "file. Write it repo-relative.")

    tl = tgt.read_text(encoding="utf-8", errors="replace").splitlines()
    assert hi <= len(tl), (
        f"{where} cites `{target}:{lo}-{hi}` but that file has only {len(tl)} lines. "
        "The pointer is stale.")

    span = tl[lo - 1:hi]
    body = "\n".join(span)

    idents = [s for s in _IDENT.findall(ctx)
              if s not in _NOISE and not _CITE.search(s) and len(s) > 2
              and _LOOKS_LIKE_CODE.match(s)]
    # Quoted claims are read POSTFIX ONLY -- from the text after this pointer, up to the
    # next one. English puts the quote after the citation ("pipeline.py:211 now reads X"),
    # so a quote sitting BEFORE a pointer almost always belongs to the pointer before it:
    # `pipeline.py:956` was inheriting `pipeline.py:211`'s quotation and failing on it.
    idents += _QUOTED.findall(after)
    if idents:
        scope = ""
        if tgt.suffix == ".py":
            scope = " ".join(_enclosing_scopes(tl, lo))
        low = (body + "\n" + scope).lower()
        def _present(s: str) -> bool:
            s = s.strip("()").strip()
            # `Spec.dc_gain_db_min` should match the field's own declaration, and prose
            # `solved` should match the constant `SOLVED`.
            return s.lower() in low or s.rsplit(".", 1)[-1].lower() in low
        assert any(_present(i) for i in idents), (
            f"{where} cites `{target}:{lo}-{hi}` while naming {idents}, but none of "
            f"those appear at that location. Lines {lo}-{hi} are:\n"
            f"{body}\n"
            "Either the pointer drifted (fix the number) or the sentence describes "
            "something that is no longer there (fix the sentence). Do NOT relax this "
            "test: a pointer that lands elsewhere discredits a claim that may be true.")
    else:
        assert body.strip() not in ("", "#", '"""'), (
            f"{where} cites `{target}:{lo}-{hi}`, which is blank or a bare delimiter, "
            "and the citing sentence names no `identifier` to check it against. Add a "
            "backtick-quoted symbol to the sentence so this is checkable, or point at "
            "the line that carries the thing being claimed.")


def _attributions(ctx: str, after: str) -> list[str]:
    """Symbols/quotations this citation can be checked against. Empty == uncheckable."""
    out = [s for s in _IDENT.findall(ctx)
           if s not in _NOISE and not _CITE.search(s) and len(s) > 2
           and _LOOKS_LIKE_CODE.match(s)]
    return out + _QUOTED.findall(after)


#: Citations naming nothing checkable, as measured 07 Sep 2026. THIS NUMBER MAY FALL AND
#: MUST NOT RISE. It is the honest size of this file's blind spot: each one is a pointer
#: held only to "lands on a non-blank line", which is the shape the README's area-supremum
#: citation had -- stale for a day, and this suite would have passed it.
MAX_UNATTRIBUTABLE = 21


def test_uncheckable_citations_do_not_multiply():
    """A ratchet, because the blind spot is real and silent.

    The fix for any one of these is to name a `symbol` in the sentence, or to quote what
    the line reads. Both make the pointer checkable forever. Lowering the constant when
    that happens is the point; raising it re-opens the hole this file was written to close.
    """
    bad = [f"{c[0].relative_to(ROOT)}:{c[1]} -> {c[4]}:{c[5]}"
           for c in CITATIONS if not _attributions(c[2], c[3])]
    assert len(bad) <= MAX_UNATTRIBUTABLE, (
        f"{len(bad)} citations name nothing checkable, up from {MAX_UNATTRIBUTABLE}. "
        "New pointers must name a `symbol` or quote what the line reads:\n  "
        + "\n  ".join(sorted(bad)))


def test_attribution_window_excludes_the_citation_itself():
    """The scanner's own off-by-one, caught and pinned.

    The window of text a citation draws its symbols from originally ran to `lines[:i]`,
    which INCLUDES the citing line. Every citation then matched itself as its own
    predecessor, `ctx` came back empty, and all 36 pointers silently fell through to the
    weak non-blank check -- a test that passes while checking nothing. It was found by
    counting attributions, not by a failure, because there was no failure to see.

    So: a synthetic line whose symbol sits before its pointer must still be attributed.
    """
    # Split so the fixture's own fake pointer is not a citation IN THIS FILE -- the
    # scanner reads its own source, and a literal here would be collected and checked.
    src = ["irrelevant preamble",
           "the `SENTINEL_SYMBOL` lives at (`src/eqrl/pipeline." + "py:1`) and nowhere"]
    lines_seen = []
    for i, ln in enumerate(src, 1):
        for m in _CITE.finditer(ln):
            window = "\n".join(src[max(0, i - 3):i - 1]) + "\n" + ln[:m.start()]
            lines_seen.append(window)
    assert lines_seen, "the fixture stopped containing a citation"
    assert "SENTINEL_SYMBOL" in lines_seen[0], (
        "the attribution window no longer sees a symbol that precedes its own pointer on "
        "the same line. Every citation is now unattributable and this suite checks "
        "nothing. Compare the slice against lines[max(0, i - 3):i - 1].")
