"""Regenerate site/index.html from web/index.html.

`web/` is the deployed page: a normal HTML document plus `eye.png` beside it.
`site/index.html` is the standalone copy — same body, no `<head>`/`<body>` wrapper, and
the figure inlined as a data URI so the file opens with nothing else next to it.

The two were previously kept in step by hand, which is how they drift. This script makes
web/ the single source and site/ a build product.

    python scripts/build_site.py            # rebuild
    python scripts/build_site.py --check    # exit 1 if site/ is stale
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "web" / "index.html"
IMG = ROOT / "web" / "eye.png"
DST = ROOT / "site" / "index.html"


def build() -> str:
    html = SRC.read_text(encoding="utf-8")

    # Strip only the document wrapper -- the doctype, the <html>/<head>/<body> tags and
    # the <meta> lines. <title> and the whole <style> block STAY: they live in the head
    # but the standalone file needs them, and dropping everything up to </head> silently
    # deletes the stylesheet.
    lines = [l for l in html.splitlines()
             if not re.match(r"\s*(<!doctype|</?html|</?head>|</?body>|<meta\b)", l, re.I)]
    body = "\n".join(lines).strip("\n") + "\n"

    uri = "data:image/png;base64," + base64.b64encode(IMG.read_bytes()).decode("ascii")
    body, n = re.subn(r'src="eye\.png"', 'src="%s"' % uri, body)
    if n != 1:
        raise SystemExit("expected exactly one eye.png reference in web/index.html, "
                         "found %d — the inlining rule needs updating." % n)
    return body


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    out = build()
    if args.check:
        if not DST.exists() or DST.read_text(encoding="utf-8") != out:
            print("STALE: site/index.html does not match web/. Run scripts/build_site.py")
            sys.exit(1)
        print("site/index.html is up to date with web/")
        return
    DST.write_text(out, encoding="utf-8")
    print("site/index.html <- web/index.html + web/eye.png  (%.0f KB)"
          % (len(out.encode("utf-8")) / 1024))


if __name__ == "__main__":
    main()
