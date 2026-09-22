"""Generate the GitHub social-preview card (1200x630). Reproducible: python3 make_social_preview.py

What makes this a map rather than an awesome-list is that each entry records the limitation its
own authors wrote down. The card shows four of those, one line each, with the section they come
from. The repository table is parsed at draw time so the count cannot drift.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cardkit import SANS, card  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
n_repos = len(re.findall(r"^\| \[", README, re.M))

# each pair is a system and the limitation its own authors state, sourced in the section named
STATED = [("V-JEPA 2-AC", "planning is run at horizon 1"),
          ("DINO-WM", "no reward signal, no task transfer"),
          ("TD-MPC2", "needs retraining per embodiment"),
          ("DreamerV3", "one hyperparameter set, at a cost")]
for name, _ in STATED:
    if name.split()[0] not in README:
        raise SystemExit(f"{name} is no longer in the map; the card would misrepresent it")


def chart(ax, accent):
    import matplotlib.pyplot as plt
    ax.text(0.78, 3.42, f"{n_repos} systems, each with what its authors admit",
            fontsize=34, color="#55585c", family=SANS)
    y = 2.72
    for name, limit in STATED:
        ax.add_patch(plt.Rectangle((0.80, y - 0.17), 0.30, 0.34, color=accent, zorder=3))
        ax.text(1.32, y, name, fontsize=34, fontweight="bold", color="#17181a",
                family=SANS, va="center")
        ax.text(4.40, y, limit, fontsize=34, color="#55585c", family=SANS, va="center")
        y -= 0.60



out = card(
    out=str(pathlib.Path(__file__).parent / "social-preview.png"),
    accent="#8250df", badge="W",
    kicker="RESEARCH MAP  ·  CC BY 4.0",
    headline="Open-source world models",
    evidence="and where their authors say they break",
    chart=chart,
    footer="github.com/GuoCheng24/world-model-map",
    headline_size=46,
)
print(f"written {pathlib.Path(out).name}  {n_repos} systems in the table")
