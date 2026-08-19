"""Why nobody can tell you how far to roll out. Bound from arXiv:1906.08253."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse
from sciglyph import set_canvas, RC, report
from sciglyph.arch import flow, cuboid, BLUE

plt.rcParams.update(RC)
fig = plt.figure(figsize=(12.4, 5.6), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
AR = set_canvas(fig)
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

# ============================================================
# Lower half: why the standard conformal recipe does not reach this.
# Drawn rather than described - the obstacle is a property of the
# dependency structure, so the dependency structure is what is shown.
# ============================================================
ax.text(.055, .560, "the obstacle, drawn", fontsize=8.4, weight="bold", color=INK, zorder=20)
ax.text(.055, .528,
        "split conformal needs exchangeability: any reordering of the points must leave the joint distribution unchanged.",
        fontsize=7.2, color=MUTE, zorder=20)

def dot(x, y, lab, c, r=.0155, fs=7.6, z=10):
    ax.add_patch(Ellipse((x, y), 2 * r / AR, 2 * r, fc="white", ec=c, lw=1.2, zorder=z))
    ax.text(x, y, lab, fontsize=fs, ha="center", va="center", color=c, zorder=z + 1)

# ---- left: i.i.d. calibration set, a swap changes nothing ----
LX, LY = .105, .355
ax.text(LX - .040, LY + .128, "calibration set   (i.i.d. draws)",
        fontsize=7.4, weight="bold", color=GREEN, zorder=20)
xs = [LX + k * .086 for k in range(4)]
for k, x in enumerate(xs):
    dot(x, LY, "$s_%d$" % (k + 1), GREEN)
ax.annotate("", xy=(xs[3], LY + .030), xytext=(xs[1], LY + .030),
            arrowprops=dict(arrowstyle="<|-|>", color=GREEN, lw=1.0,
                            connectionstyle="arc3,rad=-0.5"), zorder=8)
ax.text((xs[1] + xs[3]) / 2, LY + .092, "swap", fontsize=6.8, ha="center", color=GREEN, zorder=20)
ax.text(LX - .040, LY - .058,
        "no arrows to break. the points do not depend on each\n"
        "other, so every ordering is equally likely and the\n"
        "calibration quantile is valid.",
        fontsize=7.0, va="top", color=MUTE, zorder=20, linespacing=1.75)

# ---- right: a rollout, where the same swap is illegal ----
RX, RY = .590, .355
ax.text(RX - .040, RY + .128, "rollout   (each step made from the last)",
        fontsize=7.4, weight="bold", color=RED, zorder=20)
xs2 = [RX + k * .086 for k in range(4)]
for k, x in enumerate(xs2):
    dot(x, RY, "$s_%d$" % (k + 1), RED)
for k in range(3):
    flow(ax, (xs2[k] + .018, RY), (xs2[k + 1] - .018, RY), c=RED, lw=1.1, ms=8, z=9)
ax.annotate("", xy=(xs2[3], RY + .030), xytext=(xs2[1], RY + .030),
            arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=1.0,
                            connectionstyle="arc3,rad=-0.5"), zorder=8)
mx = (xs2[1] + xs2[3]) / 2
ax.text(mx, RY + .092, "same swap", fontsize=6.8, ha="center", color=RED, zorder=20)
ax.plot([mx - .013, mx + .013], [RY + .050, RY + .076], color=RED, lw=1.9, zorder=14)
ax.plot([mx - .013, mx + .013], [RY + .076, RY + .050], color=RED, lw=1.9, zorder=14)
ax.text(RX - .040, RY - .058,
        "the arrows are the joint distribution. reorder these and\n"
        "you have written down a different process, so the\n"
        "exchangeability argument never starts.",
        fontsize=7.0, va="top", color=MUTE, zorder=20, linespacing=1.75)

# ---- what a usable result would have to be ----
box(.055, .095, .890, .088, "#f3f8f4", GREEN, 1.2)
ax.text(.072, .146, "so the way through is not split conformal applied to rollouts",
        fontsize=7.8, weight="bold", color="#1f5c39", zorder=20)
ax.text(.072, .116,
        "it has to survive the dependence itself: a finite-sample bound at horizon k, for a specific rollout, "
        "with no assumption on the model. open problem, not a recipe.",
        fontsize=7.2, color="#2e7d4f", zorder=20)

ax.text(.5, .040,
        "MBPO bound: arXiv:1906.08253 Thm 4.1   |   ensemble approach: arXiv:2105.05716   |   "
        "the error cone is illustrative; the dependency graphs are definitions",
        fontsize=6.5, ha="center", color=MUTE, style="italic", zorder=20)

report(fig, ax)
fig.savefig(Path(__file__).with_name("rollout.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
print("saved")
