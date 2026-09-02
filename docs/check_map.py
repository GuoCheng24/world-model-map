#!/usr/bin/env python3
"""Hold the map to its own promises.

Two checks, both of which have already caught something:

1. Every bibliography entry marked as a paper carries an identifier. The map
   says "every citation was checked"; an entry with no identifier cannot have
   been, and four such entries sat in the list unnoticed.
2. The recursion numbers quoted in the prose match the closed form the figure
   script draws. A ratio quoted as 4400x was the amplification factor
   1.15^60, not the ratio between the two bounds (2927x).

Run: python docs/check_map.py   (no dependencies, no network)
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# A bibliography line looks like:  - **Title** 📖 — [arXiv:1906.08253](...)
ENTRY = re.compile(r"^- \*\*(?P<title>[^*]+)\*\*\s*(?P<mark>📖|📄)(?P<rest>.*)$", re.M)
IDENT = re.compile(r"arXiv:\d{4}\.\d{4,5}|10\.\d{4,9}/\S+|openreview\.net|proceedings\.|doi\.org")


def check_citations(text: str) -> list[str]:
    bad = []
    for m in ENTRY.finditer(text):
        if not IDENT.search(m.group("rest")):
            bad.append(m.group("title").strip())
    return bad


def bound(L: float, k: int = 60, delta: float = 0.01) -> float:
    """e_k for e_{k+1} = L*e_k + delta with e_0 = 0 - the form docs/recursion_figure.py draws."""
    return delta * k if L == 1 else delta * (L ** k - 1) / (L - 1)


def check_recursion(text: str) -> list[str]:
    problems = []
    for L, want in ((0.90, "0.0998"), (1.00, "0.60"), (1.15, "292")):
        got = bound(L)
        if abs(got - float(want)) > 0.51 * 10 ** -(len(want.split(".")[1]) if "." in want else 0) * 10:
            problems.append(f"table row L={L} says {want}, closed form gives {got:.4g}")
    m = re.search(r"(\d+)% change in `L` moves the bound by \*\*([\d,]+)×\*\*", text)
    if not m:
        problems.append("the prose no longer states the L-change and the ratio")
    else:
        pct, ratio = int(m.group(1)), int(m.group(2).replace(",", ""))
        true_pct = round((1.15 - 0.90) / 0.90 * 100)
        true_ratio = round(bound(1.15) / bound(0.90))
        if pct != true_pct:
            problems.append(f"prose says a {pct}% change in L; 0.90 to 1.15 is {true_pct}%")
        if abs(ratio - true_ratio) > max(2, 0.01 * true_ratio):
            problems.append(f"prose says {ratio}x; the two bounds differ by {true_ratio}x")
    return problems


def main() -> int:
    text = README.read_text(encoding="utf-8")
    uncited = check_citations(text)
    recursion = check_recursion(text)
    total = len(ENTRY.findall(text))
    print(f"bibliography entries: {total}, without an identifier: {len(uncited)}")
    for t in uncited:
        print(f"  MISSING IDENTIFIER: {t}")
    print(f"recursion checks: {'ok' if not recursion else str(len(recursion)) + ' problem(s)'}")
    for p in recursion:
        print(f"  {p}")
    return 1 if (uncited or recursion) else 0


if __name__ == "__main__":
    sys.exit(main())
