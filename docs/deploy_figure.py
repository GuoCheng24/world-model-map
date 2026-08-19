"""Which checkpoint, and which route in. All facts from the official repos."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from sciglyph import set_canvas, RC, report
from sciglyph.arch import flow, aspect

plt.rcParams.update(RC)
fig = plt.figure(figsize=(12.6, 5.4), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
set_canvas(fig)
INK, MUTE = "#1a1a1a", "#6b6b6b"
BLUE, GREEN, ORANGE, PURPLE, RED = "#3a6a9a", "#2e7d4f", "#b8860b", "#7a5aa8", "#c0392b"

def box(x, y, w, h, fc, ec, lw=1.1, r=.012, z=4, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls))

ax.text(.5, .962, "picking a checkpoint, and getting it to run",
        fontsize=11.2, ha="center", weight="bold", color=INK, zorder=20)

# ---------------- question -> model ----------------
ax.text(.030, .900, "what do you want to do?", fontsize=8.6, ha="left",
        weight="bold", color=INK, zorder=20)
rows = [
    ("classify or retrieve video", "V-JEPA 2.1 encoder + light probe", BLUE, "frozen encoder; no dynamics involved"),
    ("predict how a scene evolves", "V-JEPA 2 encoder + predictor", GREEN, "the predictor is what makes it a world model"),
    ("plan robot actions to a goal", "V-JEPA 2-AC", ORANGE, "the only action-conditioned checkpoint"),
    ("model-based RL in a simulator", "DreamerV3  ·  TD-MPC2", PURPLE, "they train their own dynamics + RL loop"),
]
y = .820
for q, model, col, why in rows:
    box(.030, y - .052, .225, .092, "#f8f9fb", "#c9c9c9", 1.0)
    ax.text(.042, y - .006, q, fontsize=7.3, va="center", color=INK, zorder=20)
    flow(ax, (.259, y - .006), (.288, y - .006), c=col, lw=1.2, ms=9)
    box(.292, y - .052, .240, .092, "#fbfcfd", col, 1.2)
    ax.text(.304, y + .014, model, fontsize=7.5, va="center", weight="bold", color=col, zorder=20)
    ax.text(.304, y - .026, why, fontsize=6.3, va="center", color=MUTE, zorder=20)
    y -= .112

# ---------------- size axis ----------------
ax.text(.030, .318, "size drives your GPU requirement more than anything else",
        fontsize=8.6, ha="left", weight="bold", color=INK, zorder=20)
AX0, AX1, AY = .055, .530, .215
ax.plot([AX0, AX1], [AY, AY], color="#b4b4b4", lw=1.6, zorder=5,
        solid_capstyle="round")
sizes = [("ViT-B", 80, .0), ("ViT-L", 300, .28), ("ViT-H", 600, .50),
         ("ViT-g", 1000, .72), ("ViT-G", 2000, 1.0)]
for name, params, t in sizes:
    x = AX0 + t * (AX1 - AX0)
    r = .0075 + .0235 * (params / 2000) ** .62
    start = name == "ViT-B"
    ax.add_patch(Circle((x, AY), r, transform=ax.transData,
                        fc=GREEN if start else "#c9d6e4",
                        ec=GREEN if start else "#8fa8c8", lw=1.3, zorder=7))
    ax.text(x, AY + .062, name, fontsize=7.0, ha="center", weight="bold",
            color=GREEN if start else INK, zorder=20)
    ax.text(x, AY - .062, f"{params}M" if params < 1000 else f"{params//1000}B",
            fontsize=6.6, ha="center", va="top", color=MUTE, zorder=20)
ax.text(.055, .058, "green = start here: 25x smaller than ViT-G, and it still exercises the whole pipeline",
        fontsize=6.9, ha="left", color=GREEN, zorder=20)

# ---------------- three routes in ----------------
ax.text(.575, .900, "three ways in, least friction first", fontsize=8.6,
        ha="left", weight="bold", color=INK, zorder=20)
routes = [
    ("1", "Colab", GREEN, "nothing to install",
     "the repo ships a demo notebook that loads\na model and classifies a sample video"),
    ("2", "HuggingFace", BLUE, "features only",
     "facebook/vjepa2-vitl-fpc64-256 and friends;\nskips the repo's own data stack entirely"),
    ("3", "from source", ORANGE, "predictor / training",
     "conda create -n vjepa2-312 python=3.12\npip install .   +   wget the checkpoint"),
]
y = .812
for num, name, col, tag, detail in routes:
    box(.575, y - .098, .400, .146, "#fbfcfd", col, 1.2)
    ax.add_patch(Circle((.601, y - .002), .015, transform=ax.transData,
                        fc=col, ec="none", zorder=7))
    ax.text(.601, y - .002, num, fontsize=7.6, ha="center", va="center",
            color="white", weight="bold", zorder=8)
    ax.text(.626, y + .014, name, fontsize=8.2, va="center", weight="bold",
            color=col, zorder=20)
    ax.text(.626 + len(name) * .0112 + .014, y + .014, f"· {tag}", fontsize=6.5,
            va="center", color=MUTE, style="italic", zorder=20)
    ax.text(.626, y - .044, detail, fontsize=6.5, va="center", color=INK,
            zorder=20, family="monospace" if num == "3" else None)
    y -= .168

box(.575, .075, .400, .175, "#fdf3f2", RED, 1.2, ls=(0, (4, 2.5)))
ax.text(.593, .222, "two things the repo does not tell you", fontsize=7.4,
        ha="left", weight="bold", color=RED, zorder=20)
ax.text(.593, .174, "macOS — depends on decord, which does not support it and is\n"
                    "unmaintained; use route 1 or 2 there instead",
        fontsize=6.5, ha="left", va="center", color=INK, zorder=20)
ax.text(.593, .112, "VRAM — no requirement is stated anywhere. Judge from the size\n"
                    "axis, and note video models are frame-count sensitive too",
        fontsize=6.5, ha="left", va="center", color=INK, zorder=20)

ax.text(.5, .034, "checkpoint sizes and install steps taken from the official repositories; "
                  "all three download URLs were checked live",
        fontsize=6.7, ha="center", color=MUTE, style="italic", zorder=20)

report(fig, ax)
fig.savefig(Path(__file__).with_name("deploy.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
print("saved")
