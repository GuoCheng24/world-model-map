"""Why nobody can tell you how far to roll out. Bound from arXiv:1906.08253."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from sciglyph import set_canvas, RC, report
from sciglyph.arch import flow, cuboid, BLUE

plt.rcParams.update(RC)
fig = plt.figure(figsize=(12.4, 5.6), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
set_canvas(fig)
INK, MUTE = "#1a1a1a", "#6b6b6b"
RED, GREEN, AMBER, PURPLE = "#c0392b", "#2e7d4f", "#b8860b", "#7a5aa8"

def box(x, y, w, h, fc, ec, lw=1.1, r=.012, z=4, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls))

ax.text(.5, .958, "how far can you trust the rollout? nobody can currently tell you",
        fontsize=11.2, ha="center", weight="bold", color=INK, zorder=20)

# ---------- top: the error cone ----------
PX0, PY, PW = .055, .760, .560
h = np.linspace(0, 1, 200)
err = 0.02 + 0.30 * h ** 1.8
ax.fill_between(PX0 + h * PW, PY - err * .20, PY + err * .20,
                color=RED, alpha=.16, lw=0, zorder=4)
ax.plot(PX0 + h * PW, PY + err * .20, color=RED, lw=1.2, zorder=6)
ax.plot(PX0 + h * PW, PY - err * .20, color=RED, lw=1.2, zorder=6)
ax.plot([PX0, PX0 + PW], [PY, PY], color="#8a8a8a", lw=1.0, ls=(0, (3, 2)), zorder=5)
for i in range(6):
    x = PX0 + (i / 5) * PW
    cuboid(ax, x - .008, PY - .022, .014, .044, d=.012, cols=BLUE,
           alpha=max(.18, 1.0 - i * .17))
    ax.text(x, PY - .052 - (.026 if i >= 4 else 0),
            f"$\\hat{{z}}_{{t+{i}}}$" if i else "$z_t$",
            fontsize=6.6, ha="center", va="top", color=INK if i < 2 else MUTE, zorder=20)
ax.text(PX0 + PW + .014, PY, "true state\ndrifts out of\nthe prediction",
        fontsize=6.8, va="center", color=RED, zorder=20)
ax.text(PX0, PY + .098, "prediction feeds back into itself, so error compounds",
        fontsize=7.6, ha="left", color=INK, weight="bold", zorder=20)

# V-JEPA 2 marker
ax.plot([PX0 + PW / 5] * 2, [PY - .034, PY + .075], color=GREEN, lw=1.3,
        ls=(0, (3, 2)), zorder=7)
ax.text(PX0 + PW / 5 + .008, PY + .062, "V-JEPA 2 plans here (horizon 1)",
        fontsize=6.7, color=GREEN, zorder=20)

# ---------- three attempts ----------
Y = .500
cards = [
    (.055, "MBPO (2019)", RED, "a bound that exists",
     r"$\eta[\pi]\;\geq\;\hat{\eta}[\pi]-C(\epsilon_m,\epsilon_\pi)$",
     "$\\epsilon_m$ bounds the TV-distance between\ntrue and model transitions",
     "but $\\epsilon_m$ is not observable — you must\nestimate the very thing being bounded"),
    (.375, "ensembles (2021)", AMBER, "a signal that works",
     "spread of an ensemble\nof learned dynamics",
     "cheap, and it does detect trouble\nin practice",
     "spread is not coverage; nothing forces\nit to be calibrated"),
    (.695, "conformal", GREEN, "the guarantee you want",
     "distribution-free, finite-sample\ncoverage at a chosen level",
     "needs no assumption on the model\nor the error distribution",
     "but split conformal needs exchangeability —\nand rollout steps are not exchangeable"),
]
for x, name, col, tag, formula, good, catch in cards:
    box(x, Y - .215, .250, .330, "#fbfcfd", col, 1.3)
    ax.text(x + .014, Y + .082, name, fontsize=8.4, ha="left", weight="bold",
            color=col, zorder=20)
    ax.text(x + .014, Y + .046, tag, fontsize=6.6, ha="left", color=MUTE,
            style="italic", zorder=20)
    ax.add_patch(Rectangle((x + .014, Y - .020), .222, .054, fc="#f2f4f7",
                           ec="none", zorder=5))
    ax.text(x + .125, Y + .007, formula, fontsize=7.0, ha="center", va="center",
            color=INK, zorder=20)
    ax.text(x + .014, Y - .062, good, fontsize=6.4, ha="left", va="center",
            color=INK, zorder=20)
    ax.plot([x + .014, x + .236], [Y - .102, Y - .102], color="#e0e0e0", lw=.9, zorder=5)
    ax.text(x + .014, Y - .152, catch, fontsize=6.4, ha="left", va="center",
            color=col, zorder=20)

# ---------- bottom: why exchangeability fails ----------
BY = .175
box(.055, BY - .105, .560, .175, "#f7f4fb", PURPLE, 1.3)
ax.text(.072, BY + .042, "why the standard recipe does not apply",
        fontsize=7.8, ha="left", weight="bold", color=PURPLE, zorder=20)
xs = [.110, .200, .290, .380]
for i, x in enumerate(xs):
    _r = .0165
    ax.add_patch(plt.matplotlib.patches.Ellipse(
        (x, BY - .022), 2 * _r / (12.4 / 5.6), 2 * _r,
        fc="#e8e2f2", ec=PURPLE, lw=1.1, zorder=6))
    ax.text(x, BY - .022, f"$s_{i+1}$", fontsize=6.6, ha="center", va="center",
            color=INK, zorder=8)
    if i:
        ax.annotate("", xy=(x - .0090, BY - .022), xytext=(xs[i-1] + .0090, BY - .022),
                    arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.4,
                                    mutation_scale=11), zorder=9)
ax.text(.420, BY - .022, "each score is computed from\nthe previous step's output",
        fontsize=6.6, va="center", color=INK, zorder=20)
ax.text(.072, BY - .078, "exchangeability asks that any ordering be equally likely. Here the order "
                         "is the causal structure.",
        fontsize=6.6, ha="left", color=PURPLE, zorder=20)

box(.640, BY - .105, .305, .175, "#eef6f1", GREEN, 1.3)
ax.text(.657, BY + .042, "what would settle it", fontsize=7.8, ha="left",
        weight="bold", color=GREEN, zorder=20)
ax.text(.657, BY - .030, "a finite-sample statement about a\nspecific rollout at a specific horizon,\n"
                         "with no assumption on the model.",
        fontsize=6.6, ha="left", va="center", color=INK, zorder=20)
ax.text(.657, BY - .086, "open — this is not applying a recipe.",
        fontsize=6.5, ha="left", color=GREEN, style="italic", zorder=20)

ax.text(.5, .034, "MBPO bound quoted from arXiv:1906.08253 Theorem 4.1; ensemble approach "
                  "from arXiv:2105.05716; the error cone is illustrative",
        fontsize=6.6, ha="center", color=MUTE, style="italic", zorder=20)

report(fig, ax)
fig.savefig(Path(__file__).with_name("rollout.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
print("saved")
