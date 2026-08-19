"""Why a JEPA objective collapses, and what each VICReg term forbids.
Synthetic embeddings; the mechanism is from arXiv:2105.04906."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from sciglyph import set_canvas, RC, report

plt.rcParams.update(RC)
fig = plt.figure(figsize=(12.2, 4.4), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
set_canvas(fig)
INK, MUTE = "#1a1a1a", "#6b6b6b"
RED, GREEN, BLUE, PURPLE = "#c0392b", "#2e7d4f", "#3a6a9a", "#7a5aa8"
rng = np.random.default_rng(3)

ax.text(.5, .945, "the objective has a trivial optimum — everything else exists to forbid it",
        fontsize=10.6, ha="center", weight="bold", color=INK, zorder=20)

def scatter_panel(x0, y0, w, h, pts, title, colour, caption, loss=None):
    ax.add_patch(Rectangle((x0, y0), w, h, fc="white", ec="#c9c9c9", lw=1.0, zorder=4))
    ax.scatter(x0 + pts[:, 0] * w, y0 + pts[:, 1] * h, s=9, c=colour,
               alpha=.85, linewidths=0, zorder=6)
    ax.text(x0 + w / 2, y0 + h + .052, title, fontsize=8.4, ha="center",
            weight="bold", color=colour, zorder=20)
    ax.text(x0 + w / 2, y0 - .030, caption, fontsize=6.7, ha="center",
            va="top", color=MUTE, zorder=20)
    if loss:
        ax.text(x0 + w / 2, y0 + h + .016, loss, fontsize=6.8, ha="center",
                color=INK, zorder=20)

W_, H_, Y = .175, .330, .420

# 1 collapsed
p = np.clip(.5 + rng.normal(0, .012, (160, 2)), .04, .96)
scatter_panel(.030, Y, W_, H_, p, "collapsed", RED,
              "every input maps to one point;\nprediction is perfect and useless",
              "prediction loss = 0")

# 2 variance term
p = np.column_stack([np.clip(.5 + rng.normal(0, .21, 160), .04, .96),
                     np.clip(.5 + rng.normal(0, .012, 160), .04, .96)])
scatter_panel(.253, Y, W_, H_, p, "variance term", BLUE,
              "each dimension must keep spread\n$v(Z)=\\frac{1}{d}\\sum_j \\max(0,\\gamma-S(z_j))$",
              "blocks the constant solution")

# 3 still correlated
t = rng.normal(0, .21, 160)
p = np.column_stack([np.clip(.5 + t, .04, .96),
                     np.clip(.5 + t + rng.normal(0, .02, 160), .04, .96)])
scatter_panel(.476, Y, W_, H_, p, "spread, but redundant", "#b8860b",
              "dimensions duplicate each other -\nvariance alone does not forbid this",
              "off-diagonal covariance high")

# 4 decorrelated
p = np.clip(.5 + rng.normal(0, .20, (160, 2)), .04, .96)
scatter_panel(.699, Y, W_, H_, p, "+ covariance term", GREEN,
              "off-diagonals driven to zero\n$c(Z)=\\frac{1}{d}\\sum_{i\\neq j}[C(Z)]^2_{ij}$",
              "dimensions carry distinct information")

for x in (.216, .439, .662):
    ax.annotate("", xy=(x + .030, Y + H_ / 2), xytext=(x, Y + H_ / 2),
                arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.1, mutation_scale=10))

# the gradient detail
ax.add_patch(FancyBboxPatch((.030, .075), .560, .175,
                            boxstyle="round,pad=0,rounding_size=.012",
                            fc="#fdf8ec", ec="#b8860b", lw=1.2, zorder=4))
ax.text(.048, .205, "the detail that decides whether it works",
        fontsize=7.8, ha="left", weight="bold", color="#8a6510", zorder=20)
ax.text(.048, .150, "the hinge is on the standard deviation $S=\\sqrt{\\mathrm{Var}+\\epsilon}$, not the variance.",
        fontsize=7.2, ha="left", va="center", color=INK, zorder=20)
ax.text(.048, .108, "with $\\mathrm{Var}$ the gradient vanishes as $x\\to\\bar{x}$ — the term stops pushing "
                    "exactly when it is needed, and the embeddings collapse anyway.",
        fontsize=7.0, ha="left", va="center", color=INK, zorder=20)

ax.add_patch(FancyBboxPatch((.620, .075), .355, .175,
                            boxstyle="round,pad=0,rounding_size=.012",
                            fc="#f4f1f9", ec=PURPLE, lw=1.2, zorder=4))
ax.text(.638, .205, "the other route", fontsize=7.8, ha="left", weight="bold",
        color=PURPLE, zorder=20)
ax.text(.638, .140, "SimSiam reaches the same goal with a stop-gradient\nand a predictor head — no negative pairs, no\nmomentum encoder, and no explicit statistics.",
        fontsize=7.0, ha="left", va="center", color=INK, zorder=20)

ax.text(.5, .028, "mechanism from VICReg (arXiv:2105.04906) and SimSiam (arXiv:2011.10566); "
                  "point clouds are synthetic",
        fontsize=6.6, ha="center", color=MUTE, style="italic", zorder=20)

report(fig, ax)
fig.savefig(Path(__file__).with_name("collapse.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
print("saved")
