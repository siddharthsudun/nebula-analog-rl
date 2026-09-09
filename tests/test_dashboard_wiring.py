"""Static wiring checks for the dashboard front end.

There is one class of front-end defect that every test in this suite was blind to and
that a page load finds instantly: a panel whose `init` function is written, correct and
never called. `initCandidateGallery` shipped that way -- the gallery endpoints were
covered by `tests/test_gallery_endpoints.py`, the route returned the right payload, and
the section still sat on its loading spinner forever, because nothing in the boot
sequence invoked it. A server-side test cannot see that, and a browser test needs a
browser. A parse of the delivered JavaScript can.

These are deliberately static and cheap: no node, no browser, no network. They check the
two joins where "written" and "reachable" come apart -- a function that is defined but
never called, and an element the script addresses that the document never contains.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "dashboard" / "static"
SCRIPTS = ("app.js", "charts.js")

#: `function initFoo(` / `async function initFoo(` at any indentation.
_INIT_DEF = re.compile(r"^(?:async\s+)?function\s+(init[A-Za-z0-9_]*)\s*\(", re.M)
#: An element the script looks up by literal id, e.g. `$("#gallery-grid")`.
_ID_REF = re.compile(r'\$\("#([A-Za-z0-9_-]+)"')
_ID_IN_MARKUP = re.compile(r'\bid="([A-Za-z0-9_-]+)"')


def _script(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _all_script_text() -> str:
    return "\n".join(_script(name) for name in SCRIPTS)


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_init_function_is_actually_called(name):
    """A defined-but-never-called panel initializer is a permanently loading section."""
    text = _script(name)
    everything = _all_script_text()
    unused = []
    for func in sorted(set(_INIT_DEF.findall(text))):
        # One match is the definition itself; anything beyond it is a call or a reference.
        if len(re.findall(rf"\b{func}\s*\(", everything)) <= 1:
            unused.append(func)
    assert not unused, (
        f"{name} defines these initializers and never calls them: {', '.join(unused)}.\n"
        "Wire each one into the DOMContentLoaded handler with its own `.catch` that "
        "writes an error banner into the panel it owns. A panel whose init never runs "
        "does not fail loudly -- it shows its loading placeholder forever, which reads "
        "as a slow server rather than as a bug.")


def test_the_boot_handler_owns_every_panel_initializer():
    """The initializers must be reachable from boot, not only from each other."""
    text = _script("app.js")
    start = text.index('document.addEventListener("DOMContentLoaded"')
    boot = text[start:]
    defined = set(_INIT_DEF.findall(text))
    # `initComposer`/`initHistory` are deliberately chained off `initSpec`, because they
    # read the defaults it fetches. Everything else is booted directly.
    chained = {"initComposer", "initHistory"}
    missing = sorted(f for f in defined - chained if f not in boot)
    assert not missing, (
        f"these initializers are never reached from the boot handler: {', '.join(missing)}")


def test_every_element_the_script_addresses_exists_in_the_document():
    """`$("#id")` on a missing element returns null and throws on first use."""
    html = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    in_document = set(_ID_IN_MARKUP.findall(html))
    scripts = _all_script_text()
    # Ids the scripts create themselves, in template literals or element factories.
    injected = set(_ID_IN_MARKUP.findall(scripts))
    dangling = sorted(set(_ID_REF.findall(scripts)) - in_document - injected)
    assert not dangling, (
        "the dashboard scripts look up these ids, which index.html does not define and "
        f"no script injects: {', '.join(dangling)}.\n"
        "Either the markup lost the element or the selector has a typo; both surface at "
        "runtime as a null dereference in whichever handler touched it first.")
