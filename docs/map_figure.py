"""Reproduces the figure in this directory.

Requires sciglyph:  pip install sciglyph
Run:  python map_figure.py
world-model-map figure: the pipeline, where it degrades, and which line of
work intervenes at which stage. All facts from published papers."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from sciglyph import set_canvas, RC, report
from sciglyph.arch import flow, cuboid, trapezoid, image_thumb, BLUE, GRAY

plt.rcParams.update(RC)
fig = plt.figure(figsize=(12.4, 4.8), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
set_canvas(fig)

INK, MUTE = "#1a1a1a", "#6b6b6b"
RED, GREEN, PURPLE, ORANGE = "#c0392b", "#2e7d4f", "#7a5aa8", "#b8860b"

def box(x, y, w, h, fc, ec, lw=1.0, r=.012, z=4, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls))

YM = .800          # main pipeline axis
ax.text(.5, .955, "how a latent world model runs — and where each line of work intervenes",
        fontsize=11, ha="center", weight="bold", color=INK, zorder=20)

# ---------------- pipeline ----------------
image_thumb(ax, .020, YM - .075, .052, .150, seed=4, label="observation $o_t$", fs=7)
trapezoid(ax, .095, YM - .085, .030, .170, shrink=.28, fc="#cfe0f2", label="encoder", fs=7)
flow(ax, (.076, YM), (.092, YM), c=MUTE, lw=1.1, ms=9)

cuboid(ax, .148, YM - .050, .016, .100, d=.016, cols=BLUE)
ax.text(.160, YM - .078, "$z_t$", fontsize=8.6, ha="center", va="top", color=INK, zorder=20)
flow(ax, (.128, YM), (.145, YM), c=MUTE, lw=1.1, ms=9)

# dynamics
box(.196, YM - .058, .078, .116, "#eef4ea", "#5f8f4f", 1.1)
ax.text(.235, YM + .012, "dynamics", fontsize=7.6, ha="center", weight="bold", color=INK, zorder=20)
ax.text(.235, YM - .030, r"$\hat{z}_{t+1}=f(z_t,a_t)$", fontsize=7, ha="center", color=INK, zorder=20)
flow(ax, (.170, YM), (.192, YM), c=MUTE, lw=1.1, ms=9)
# action in
box(.196, YM - .175, .078, .062, "#f6efe0", ORANGE, 1.0)
ax.text(.235, YM - .144, "action $a_t$", fontsize=7.2, ha="center", color=INK, zorder=20)
flow(ax, (.235, YM - .110), (.235, YM - .062), c=ORANGE, lw=1.0, ms=8)

# autoregressive rollout: fading blocks = growing uncertainty
xs = [.300, .345, .390, .435, .480]
for i, x in enumerate(xs):
    a = 1.0 - i * .17
    cuboid(ax, x, YM - .050, .016, .100, d=.016, cols=BLUE, alpha=a)
    ax.text(x + .012, YM - .078, f"$\\hat{{z}}_{{t+{i+1}}}$", fontsize=6.8,
            ha="center", va="top", color=INK if i < 2 else MUTE, zorder=20)
    if i:
        flow(ax, (xs[i-1] + .030, YM), (x - .003, YM), c=MUTE, lw=.9, ms=7)
flow(ax, (.278, YM), (.297, YM), c=MUTE, lw=1.1, ms=9)
ax.annotate("", xy=(.512, YM + .085), xytext=(.298, YM + .085),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.1))
ax.text(.405, YM + .100, "error accumulates with every step", fontsize=7,
        ha="center", color=RED, style="italic", zorder=20)

# planner
box(.530, YM - .058, .080, .116, "#eaf1f8", "#5f93bd", 1.1)
ax.text(.570, YM + .012, "planner", fontsize=7.6, ha="center", weight="bold", color=INK, zorder=20)
ax.text(.570, YM - .030, "CEM / MPC", fontsize=7, ha="center", color=INK, zorder=20)
flow(ax, (.512, YM), (.526, YM), c=MUTE, lw=1.1, ms=9)
flow(ax, (.612, YM), (.634, YM), c=MUTE, lw=1.1, ms=9)
ax.text(.660, YM, "action", fontsize=8, ha="center", va="center", weight="bold",
        color=INK, zorder=20)

# ---------------- error vs horizon inset ----------------
bx, by, bw, bh = .745, YM - .095, .150, .175
ax.add_patch(Rectangle((bx, by), bw, bh, fc="white", ec="#9a9a9a", lw=.8, zorder=5))
h = np.linspace(0, 1, 60)
ax.plot(bx + h * bw, by + (0.10 + 0.80 * h ** 1.7) * bh, color=RED, lw=1.5, zorder=7)
ax.plot([bx + .085 * bw] * 2, [by, by + bh], color=GREEN, lw=1.2, ls=(0, (3, 2)), zorder=7)
ax.text(bx + .10 * bw, by + bh + .016, "V-JEPA 2 plans here", fontsize=6.4,
        ha="left", color=GREEN, zorder=20)
ax.text(bx + bw / 2, by - .020, "planning horizon", fontsize=6.6, ha="center",
        va="top", color=INK, zorder=20)
ax.text(bx - .008, by + bh / 2, "prediction\nerror", fontsize=6.4, rotation=90,
        ha="right", va="center", color=INK, zorder=20)
ax.text(bx + bw / 2, by - .052, "their own results run at horizon = 1", fontsize=6.5,
        ha="center", va="top", color=MUTE, style="italic", zorder=20)

# ---------------- lines of work, anchored under the stage they act on ------
STAGES = [(.093, .172), (.194, .278), (.297, .614)]
for x0, x1 in STAGES:                       # guide rails down from the pipeline
    for xe in (x0, x1):
        ax.plot([xe, xe], [.300, YM - .105], color="#e2e2e2", lw=.8,
                ls=(0, (2, 3)), zorder=1)

ax.text(.045, .560, "which stage each line of work acts on",
        fontsize=8.6, ha="left", weight="bold", color=INK, zorder=20)

spans = [
    (.093, .172, "representation collapse", PURPLE, True,
     "three 2026 papers - theory moving fastest"),
    (.194, .278, "action-latent geometry", ORANGE, True,
     "V-JEPA 2 calls the action frame ill-defined"),
    (.194, .278, "constrained latent dynamics", GREEN, True,
     "Koopman-style spectral limits on $f$"),
    (.297, .614, "when to trust a rollout", RED, False,
     "MBPO's bound needs an unobservable model error"),
]
yy = .500
for x0, x1, name, col, retrain, note in spans:
    ax.add_patch(FancyBboxPatch((x0, yy - .016), x1 - x0, .032,
                                boxstyle="round,pad=0,rounding_size=.008",
                                fc=col, ec="none", alpha=.20 if retrain else .40, zorder=5))
    ax.plot([x0, x1], [yy, yy], color=col, lw=2.6, solid_capstyle="butt", zorder=7)
    for xe in (x0, x1):
        ax.plot([xe, xe], [yy - .015, yy + .015], color=col, lw=1.8, zorder=7)
    ax.text(x1 + .014, yy + .006, name, fontsize=7.4, va="center",
            weight="bold", color=col, zorder=20)
    ax.text(x1 + .014, yy - .016, note, fontsize=6.3, va="center", color=MUTE, zorder=20)
    ax.text(x0 - .010, yy, "retrain" if retrain else "no retraining", fontsize=6.2,
            ha="right", va="center", color=col, style="italic", zorder=20)
    yy -= .062

ax.text(.045, .185, "faded band = changes the training objective    ·    "
                    "solid band = wraps the existing model",
        fontsize=6.9, ha="left", color=MUTE, style="italic", zorder=20)
ax.text(.045, .140, "pipeline and limitations quoted from the models' own papers; "
                    "every entry in the repository is marked full-text or abstract-only",
        fontsize=6.9, ha="left", color=MUTE, style="italic", zorder=20)

report(fig, ax)
fig.savefig(Path(__file__).with_name("map.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
print("saved")
