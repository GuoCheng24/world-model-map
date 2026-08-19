"""Which of these do you actually need, and what does each one cost you.

Everything on this chart is from the repository README, the papers, or the
checkpoint tables - nothing is inferred. Parameter counts are the released
checkpoint sizes; "needs retraining" means adopting it changes your training
objective rather than adding a module.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from sciglyph import set_canvas, RC, report
from sciglyph.arch import flow
from sciglyph._canvas import circle

plt.rcParams.update(RC)
fig = plt.figure(figsize=(12.6, 6.9), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
AR = set_canvas(fig)

INK, MUTE, RULE = "#1a1a1a", "#6b6b6b", "#c9c4bb"
BLUE, GREEN, AMBER, RED, PURPLE = "#2f6f9f", "#2e7d4f", "#b8860b", "#c0392b", "#7a5aa8"

def box(x, y, w, h, fc, ec, lw=1.2, r=.009, z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))

ax.text(.5, .963, "start from what you need it to do, not from which model is newest",
        fontsize=11.4, ha="center", weight="bold", color=INK, zorder=30)

# ---------- the question column ----------
QX, QW = .030, .225
rows = [
    ("classify or retrieve video,\nor use the features downstream",
     "V-JEPA 2.1 encoder", "80M – 2B", "frozen encoder + a light probe;\nno dynamics model involved",
     "no retraining", GREEN),
    ("predict how a scene evolves\nin representation space",
     "V-JEPA 2 predictor", "300M – 1B", "the predictor is what makes it a\nworld model rather than an encoder",
     "no retraining", GREEN),
    ("plan robot actions\nfrom a goal image",
     "V-JEPA 2-AC", "1B", "the only action-conditioned checkpoint;\nauthors plan at horizon 1",
     "no retraining,\nbut horizon 1", AMBER),
    ("model-based RL\ninside a simulator",
     "DreamerV3  ·  TD-MPC2", "varies", "they train their own latent dynamics\nand ship the RL loop",
     "trains from scratch", AMBER),
    ("plan from frozen features,\nno video pre-training",
     "DINO-WM", "ViT-B/L", "builds the world model on top of\nfrozen DINO features",
     "trains the dynamics", AMBER),
    ("bound the error of a\nlong rollout",
     "Koopman Dreamer  ·  SD-GWM", "—", "spectral radius bound, or projection\nonto a feasible set",
     "needs retraining", RED),
]
top, rh = .868, .132
for i, (q, model, size, why, cost, col) in enumerate(rows):
    y = top - i * rh
    ax.text(QX, y - .030, q, fontsize=8.0, va="center", color=INK, zorder=20, linespacing=1.5)
    flow(ax, (QX + QW, y - .030), (QX + QW + .036, y - .030), c=MUTE, lw=1.0, ms=8)

    mx = QX + QW + .046
    box(mx, y - .062, .208, .064, "#f4f7fa", BLUE, 1.1)
    ax.text(mx + .104, y - .022, model, fontsize=8.4, ha="center", color=INK,
            weight="bold", zorder=20)
    ax.text(mx + .104, y - .046, size, fontsize=7.0, ha="center", color=MUTE, zorder=20)

    ax.text(mx + .228, y - .030, why, fontsize=7.4, va="center", color=MUTE,
            zorder=20, linespacing=1.55)

    cx = .905
    circle(ax, (cx, y - .030), .0088, fc=col, ec="none", zorder=8)
    ax.text(cx + .016, y - .030, cost, fontsize=7.4, va="center", color=col, zorder=20)
    if i < len(rows) - 1:
        ax.plot([QX, .975], [y - .078] * 2, color=RULE, lw=.6, zorder=3)

ax.text(QX, .922, "you want to…", fontsize=7.6, color=MUTE, zorder=20)
ax.text(QX + QW + .150, .922, "use", fontsize=7.6, ha="center", color=MUTE, zorder=20)
ax.text(QX + QW + .274, .922, "because", fontsize=7.6, color=MUTE, zorder=20)
ax.text(.905, .922, "what it costs you", fontsize=7.6, color=MUTE, zorder=20)
ax.plot([QX, .975], [.906] * 2, color=INK, lw=.9, zorder=4)

# ---------- the one line most people need ----------
box(.030, .022, .945, .062, "#eef5ef", GREEN, 1.3)
ax.text(.048, .053, "If you are just starting: V-JEPA 2.1 ViT-B/16, 80M.",
        fontsize=8.8, weight="bold", color="#1f5c39", zorder=20)
ax.text(.048, .033, "It is the smallest thing that exercises the whole pipeline and over an order of "
                    "magnitude below ViT-G. Get the loop working, then scale.",
        fontsize=7.8, color="#2e7d4f", zorder=20)

report(fig, ax)
out = Path(__file__).parent / "choose.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out)
