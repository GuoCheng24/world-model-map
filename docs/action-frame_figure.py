"""Why 'infer the action frame from pixels' can be ill-posed.
The limitation is quoted from V-JEPA 2 (arXiv:2506.09985) section 4.3."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse
from sciglyph import set_canvas, RC, report

plt.rcParams.update(RC)
FW, FH = 12.2, 3.72
fig = plt.figure(figsize=(FW, FH), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
set_canvas(fig)
AR = FW / FH
INK, MUTE = "#1a1a1a", "#6b6b6b"
GREEN, RED, BLUE, ORANGE = "#2e7d4f", "#c0392b", "#3a6a9a", "#b8860b"

def box(x, y, w, h, fc, ec, lw=1.1, r=.012, z=4, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls))

def disc(x, y, r, **kw):
    ax.add_patch(Ellipse((x, y), 2 * r / AR, 2 * r, **kw))

def frame_axes(x, y, ang, L=.042, col=BLUE, lw=1.7, z=9, labels=True):
    """Draw a 2-axis coordinate frame rotated by ang (radians)."""
    for k, (dx, dy, lab) in enumerate([(np.cos(ang), np.sin(ang), "x"),
                                       (-np.sin(ang), np.cos(ang), "y")]):
        ax.annotate("", xy=(x + dx * L / AR, y + dy * L),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=9),
                    zorder=z)
        if labels:
            ax.text(x + dx * L * 1.30 / AR, y + dy * L * 1.30, lab, fontsize=6.6,
                    ha="center", va="center", color=col, zorder=z + 1)

def scene(x0, y0, w, h, show_base, title, tcol, caption):
    ax.add_patch(Rectangle((x0, y0), w, h, fc="#14161c", ec="#5a5f6b", lw=1.2, zorder=4))
    ax.text(x0 + w / 2, y0 + h + .040, title, fontsize=8.2, ha="center",
            weight="bold", color=tcol, zorder=20)
    # table
    ax.add_patch(Rectangle((x0 + .012, y0 + .030), w - .024, .052,
                           fc="#2a2f3a", ec="none", zorder=5))
    # gripper + cube
    gx, gy = x0 + w * .58, y0 + h * .58
    ax.add_patch(Rectangle((gx - .011, gy), .022, .030, fc="#c9d6e4",
                           ec="#8fa8c8", lw=.8, zorder=7))
    ax.add_patch(Rectangle((gx - .014, gy - .012), .006, .014, fc="#c9d6e4", ec="none", zorder=7))
    ax.add_patch(Rectangle((gx + .008, gy - .012), .006, .014, fc="#c9d6e4", ec="none", zorder=7))
    ax.add_patch(Rectangle((x0 + w * .30, y0 + .082), .026, .026,
                           fc="#e08a72", ec="#c0603f", lw=.8, zorder=7))
    if show_base:
        ax.add_patch(Rectangle((x0 + w * .78, y0 + .030), .042, .105,
                               fc="#3d4657", ec="#7c879b", lw=1.0, zorder=6))
        ax.text(x0 + w * .78 + .021, y0 + .022, "base", fontsize=6.2, ha="center",
                va="top", color="#9aa4b5", zorder=20)
    ax.text(x0 + w / 2, y0 - .030, caption, fontsize=6.7, ha="center", va="top",
            color=MUTE, zorder=20)
    return gx, gy

ax.text(.5, .955, "\"the problem of inferring the action coordinate axis is not well defined\"",
        fontsize=10.4, ha="center", weight="bold", color=INK, zorder=20)
ax.text(.5, .908, "V-JEPA 2-AC takes a Cartesian end-effector action but has no camera calibration — "
                  "so the frame must come from the pixels",
        fontsize=7.2, ha="center", color=MUTE, zorder=20)

SY, SH, SW = .530, .250, .250

# --- left: base visible -> identifiable ---
gx, gy = scene(.045, SY, SW, SH, True, "base in frame", GREEN,
               "the frame is pinned to something visible")
frame_axes(gx, gy + .034, 0.0, col=GREEN)
box(.045, .330, SW, .120, "#eef6f1", GREEN, 1.2)
ax.text(.045 + SW / 2, .408, "identifiable", fontsize=7.8, ha="center",
        weight="bold", color=GREEN, zorder=20)
ax.text(.045 + SW / 2, .366, "one frame explains the observation;\n"
                             "an action maps to one motion",
        fontsize=6.6, ha="center", va="center", color=INK, zorder=20)

# --- right: base out of frame -> two rival explanations ---
gx2, gy2 = scene(.360, SY, SW, SH, False, "base out of frame", RED,
                 "nothing anchors the axes")
frame_axes(gx2, gy2 + .034, 0.0, col=BLUE)
frame_axes(gx2, gy2 + .034, 0.85, col=ORANGE)
ax.text(.372, SY + SH - .030, "two frames,\nsame pixels", fontsize=6.5,
        ha="left", va="top", color="#d8dde6", zorder=20)
box(.360, .330, SW, .120, "#fdf3f2", RED, 1.2, ls=(0, (4, 2.5)))
ax.text(.360 + SW / 2, .408, "not identifiable", fontsize=7.8, ha="center",
        weight="bold", color=RED, zorder=20)
ax.text(.360 + SW / 2, .366, "two different frames predict the same\n"
                             "image — no data distinguishes them",
        fontsize=6.6, ha="center", va="center", color=INK, zorder=20)

# --- consequence ---
box(.675, .330, .282, .450, "#f8f9fb", "#9aa4b5", 1.2)
ax.text(.692, .742, "why this is a geometry problem,", fontsize=7.8,
        ha="left", weight="bold", color=INK, zorder=20)
ax.text(.692, .710, "not a training problem", fontsize=7.8,
        ha="left", weight="bold", color=INK, zorder=20)
bullets = [
    ("more data does not help", "both frames fit every example equally well"),
    ("bigger models do not help", "the information is absent from the input"),
    ("the authors' workaround", "try camera positions by hand until one works"),
]
y = .640
for head, sub in bullets:
    disc(.706, y + .006, .0055, fc="#9aa4b5", ec="none", zorder=8)
    ax.text(.722, y + .006, head, fontsize=7.0, ha="left", va="center",
            weight="bold", color=INK, zorder=20)
    ax.text(.722, y - .030, sub, fontsize=6.4, ha="left", va="center",
            color=MUTE, zorder=20)
    y -= .082
ax.plot([.692, .940], [.408, .408], color="#dcdcdc", lw=.9, zorder=5)
ax.text(.692, .372, "open: what conditions on the observation\nmake the frame identifiable at all?",
        fontsize=6.8, ha="left", va="center", color=BLUE, zorder=20)

ax.text(.5, .262, "an action is only meaningful relative to a frame — if the frame is ambiguous, "
                  "the world model is being asked an ill-posed question",
        fontsize=7.4, ha="center", color=INK, zorder=20)
ax.text(.5, .200, "limitation quoted from V-JEPA 2, arXiv:2506.09985 §4.3; the scenes are schematic",
        fontsize=6.6, ha="center", color=MUTE, style="italic", zorder=20)

report(fig, ax)
# 内容(含最下面那行脚注)只占到轴坐标 y≈.17 以上。bbox_inches="tight" 裁不掉轴*内部*的空白
# (轴本身铺满整幅), 所以直接给出以英寸计的裁剪框, 原点在左下角。
fig.savefig(Path(__file__).with_name("action-frame.png"),
            dpi=200, bbox_inches=Bbox([[0, FH * 0.170], [FW, FH]]), facecolor="white")
print("saved")
