"""Check local Markdown links in the project documentation (no network required)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^\n)]+)\)|!\[[^\]\n]*\]\(([^\n)]+)\)")


def check_links() -> list[str]:
    bad: list[str] = []
    files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    for page in files:
        for match in MARKDOWN_LINK.finditer(page.read_text(encoding="utf-8")):
            target = next(value for value in match.groups() if value is not None)
            # Our Markdown links don't contain whitespace-delimited titles.
            target = target.strip().strip("<>")
            if not target or target.startswith("#"):
                continue
            split = urlsplit(target)
            if split.scheme or split.netloc:
                continue
            path = (page.parent / unquote(split.path)).resolve()
            if not path.exists() or not path.is_relative_to(ROOT.resolve()):
                bad.append(f"{page.relative_to(ROOT)}: missing or unsafe link {target}")
    return bad


def main() -> int:
    bad = check_links()
    for item in bad:
        print(item, file=sys.stderr)
    if bad:
        return 1
    print("Markdown local link check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
