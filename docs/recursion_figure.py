"""The one recursion every rollout method is attacking, and which term each one touches.

e_{k+1} <= L * e_k + delta   ->   e_k = delta * (L^k - 1) / (L - 1)

Koopman Dreamer's bound (arXiv:2607.19719) has exactly this shape: the abstract
says it "separates amplification by the spectral backbone and bilinear interaction
from the additive effects of stochastic-state mismatch and modeling residuals".
Curves below are computed, not drawn by hand; the closed form was checked against
direct iteration to machine precision.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from sciglyph import set_canvas, RC, report
from sciglyph.arch import flow, op_circle
from sciglyph._canvas import circle

plt.rcParams.update(RC)
fig = plt.figure(figsize=(12.4, 7.0), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
AR = set_canvas(fig)
INK, MUTE, RULE = "#1a1a1a", "#6b6b6b", "#c9c4bb"
RED, GREEN, AMBER, PURPLE, BLUE = "#c0392b", "#2e7d4f", "#b8860b", "#7a5aa8", "#2f6f9f"

def box(x, y, w, h, fc, ec, lw=1.1, r=.010, z=4, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls))

ax.text(.5, .963, "every method here attacks the same recursion — they just touch different terms",
        fontsize=11.4, ha="center", weight="bold", color=INK, zorder=30)

# ============ A: the recursion itself ============
ax.text(.035, .900, "the recursion", fontsize=8.8, weight="bold", color=INK, zorder=30)
ay = .762
box(.035, ay, .062, .070, "#eef2f6", BLUE, 1.2)
ax.text(.066, ay + .035, r"$e_k$", fontsize=9.6, ha="center", va="center", zorder=10)

flow(ax, (.100, ay + .035), (.150, ay + .035), c=INK, lw=1.1)
ax.text(.125, ay + .052, r"$\times\, L$", fontsize=8.6, ha="center", color=RED, zorder=30)

# 用库里的 op_circle: 直接 Circle 在非方画布上会压成椭圆
op_circle(ax, .171, ay + .035, sym="+", r=.013, fs=8.5, z=8)

ax.annotate("", xy=(.171, ay + .020), xytext=(.171, ay - .052),
            arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.1), zorder=6)
ax.text(.171, ay - .068, r"$\delta$", fontsize=9.4, ha="center", va="top", color=AMBER, zorder=30)

flow(ax, (.186, ay + .035), (.236, ay + .035), c=INK, lw=1.1)
box(.236, ay, .074, .070, "#eef2f6", BLUE, 1.2)
ax.text(.273, ay + .035, r"$e_{k+1}$", fontsize=9.6, ha="center", va="center", zorder=10)

# feedback: next step's input is this step's output.
# 走方框上方一条明确的折线, 并置于方框之上 —— 之前用 arc3 且 zorder 低于方框,
# 弧被方框盖住, 只露出两端, 看着像两段碎线。
fby = ay + .112
ax.plot([.273, .273, .066, .066], [ay + .070, fby, fby, ay + .074],
        color=MUTE, lw=.9, zorder=12, solid_joinstyle="round")
ax.annotate("", xy=(.066, ay + .070), xytext=(.066, ay + .082),
            arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=.9), zorder=12)
ax.text(.170, fby + .010, "fed back in", fontsize=7.4, ha="center", color=MUTE, zorder=30)

ax.text(.035, .700, r"$e_k \;=\; \delta\,\dfrac{L^{k}-1}{L-1}$"
                    "\n" r"$L<1:\; e_k \to \delta/(1-L)$",
        fontsize=8.6, ha="left", va="top", color=INK, zorder=30)
ax.text(.035, .598, "L  multiplicative — how much the map stretches\n"
                    "δ  additive — new error injected every step",
        fontsize=7.6, ha="left", va="top", color=MUTE, zorder=30, linespacing=1.7)

# ============ B: computed curves ============
d = 0.01
Ls = [(0.85, GREEN), (0.90, GREEN), (1.00, AMBER), (1.05, RED), (1.15, RED)]
bx, by, bw, bh = .400, .590, .560, .300
ax.add_patch(Rectangle((bx, by), bw, bh, fc="white", ec=RULE, lw=.9, zorder=3))
K = 60
ks = np.arange(1, K + 1)
ymin, ymax = 5e-3, 6e2
def X(k): return bx + (k - 1) / (K - 1) * bw
def Y(v): return by + (np.log10(np.clip(v, ymin, ymax)) - np.log10(ymin)) / \
                      (np.log10(ymax) - np.log10(ymin)) * bh
for L, c in Ls:
    e = d * (L ** ks - 1) / (L - 1) if L != 1 else d * ks
    ax.plot(X(ks), Y(e), color=c, lw=1.5 if L != 1 else 1.7, zorder=6,
            alpha=.95 if L in (0.90, 1.00, 1.15) else .55)
    # L=0.85 与 0.9 的终值 (.0667 / .0998) 在对数轴上几乎同高, 末端标签会叠。
    # 把下面那条的标签沿曲线左移并下沉。
    lab = f"L = {L:g}"
    if L == 0.85:
        ax.text(X(K) - .120, Y(e[-1]) - .026, lab, fontsize=7.2, va="center", color=c, zorder=30)
    else:
        ax.text(X(K) + .006, Y(e[-1]), lab, fontsize=7.2, va="center", color=c, zorder=30)
for L, c in [(0.85, GREEN), (0.90, GREEN)]:
    ax.plot([bx, X(K)], [Y(d / (1 - L))] * 2, color=c, lw=.7, ls=(0, (2, 2)), alpha=.5, zorder=5)
for p in (-2, -1, 0, 1, 2):
    ax.plot([bx - .006, bx], [Y(10.0 ** p)] * 2, color=MUTE, lw=.7, zorder=5)
    ax.text(bx - .010, Y(10.0 ** p), r"$10^{%d}$" % p, fontsize=6.6, ha="right", va="center",
            color=MUTE, zorder=30)
ax.text(bx + bw / 2, by - .034, "rollout step  k", fontsize=7.6, ha="center", color=MUTE, zorder=30)
ax.text(bx - .050, by + bh / 2, "error bound", fontsize=7.6, rotation=90,
        ha="center", va="center", color=MUTE, zorder=30)
ax.text(bx + .012, by + bh - .022,
        "δ = 0.01 fixed; only L changes.\n"
        "at k = 60:  L=0.90 → 0.0998    L=1.00 → 0.60    L=1.15 → 292",
        fontsize=7.2, va="top", color=INK, zorder=30, linespacing=1.6)
ax.text(bx + bw - .012, by + .020, "a 25% change in L moves the k=60 bound by 4400×",
        fontsize=7.2, ha="right", color=RED, zorder=30)

# ============ C: which term does each method touch ============
gy_top = .500
ax.text(.035, gy_top + .022, "which term each approach actually touches",
        fontsize=8.8, weight="bold", color=INK, zorder=30)

cols = [("shrink L", .430), ("shrink δ", .545), ("project each step", .672),
        ("measure e_k", .800), ("what you end up with", .900)]
rows = [
    ("Koopman Dreamer", "arXiv:2607.19719",
     ["yes", "-", "-", "-"], "bounded-radius 2-D rotation–scaling blocks; their\nbound splits amplification from the additive terms", GREEN),
    ("SD-GWM", "arXiv:2608.08689",
     ["-", "-", "yes", "-"], "S / N fixed-form mechanisms + bounded residual R,\nthen a global projection onto feasibility", GREEN),
    ("MBPO bound", "arXiv:1906.08253",
     ["-", "yes", "-", "-"], "bounds δ through ε_m — which is not observable", AMBER),
    ("ensemble spread", "arXiv:2105.05716",
     ["-", "-", "-", "yes"], "a signal, not a bound; nothing forces calibration", AMBER),
    ("conformal (wanted)", "no such result yet",
     ["-", "-", "-", "yes"], "would certify e_k with no model assumption —\nblocked: rollout steps are not exchangeable", PURPLE),
]
rh = .074
for j, (lab, cx) in enumerate(cols):
    ax.text(cx, gy_top - .012, lab, fontsize=7.2, ha="center" if j < 4 else "left",
            color=MUTE, zorder=30)
ax.plot([.035, .965], [gy_top - .030] * 2, color=INK, lw=.9, zorder=5)

for i, (name, cite, marks, note, col) in enumerate(rows):
    y = gy_top - .046 - i * rh
    ax.text(.035, y - .012, name, fontsize=8.0, color=INK, weight="bold", zorder=30)
    ax.text(.035, y - .034, cite, fontsize=6.6, color=MUTE, zorder=30)
    for j, m in enumerate(marks):
        cx = cols[j][1]
        if m == "yes":
            # 用库里的 circle(): Circle(r/AR) 只是把圆缩小, 并没有校正长宽比, 画出来仍是椭圆
            circle(ax, (cx, y - .020), .0092, fc=col, ec="none", zorder=6)
        else:
            ax.plot([cx - .008, cx + .008], [y - .020] * 2, color=RULE, lw=1.2, zorder=6)
    ax.text(cols[4][1], y - .020, note, fontsize=6.9, va="center", color=MUTE,
            zorder=30, linespacing=1.55)
    ax.plot([.035, .965], [y - .050] * 2, color=RULE, lw=.6, zorder=4)

# ============ the gap this exposes ============
gy = gy_top - .046 - len(rows) * rh - .006
box(.035, gy - .066, .930, .062, "#f6f2fa", PURPLE, 1.2)
ax.text(.050, gy - .035,
        "the two halves of the table do not overlap: everything that constrains the recursion "
        "gives no certificate, and everything that certifies gives no constraint.",
        fontsize=8.0, va="center", color="#4a3070", zorder=30)

ax.text(.5, gy - .092,
        "recursion is the standard error-propagation form, checked against direct iteration to machine precision; "
        "Koopman Dreamer and SD-GWM read from abstracts only",
        fontsize=6.6, ha="center", va="top", color=MUTE, style="italic", zorder=30)

report(fig, ax)
out = Path(__file__).parent / "recursion.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out)
