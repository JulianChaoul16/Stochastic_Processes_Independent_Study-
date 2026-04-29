"""
Regime-Switching Time Series — GIF with Markov chain diagram
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec

# ── Parameters ────────────────────────────────────────────────────────────────
np.random.seed(7)

TRANSITION = np.array([[0.92, 0.08],
                        [0.15, 0.85]])

REGIME = {
    0: dict(mu=+0.6, sigma=0.8,  color="#22c55e", label="State 0 — Bull  ↑"),
    1: dict(mu=-0.8, sigma=2.5,  color="#ef4444", label="State 1 — Bear  ↓"),
}

N_STEPS  = 150
WINDOW   = 80

# ── Pre-generate series ───────────────────────────────────────────────────────
states    = np.empty(N_STEPS, dtype=int)
values    = np.empty(N_STEPS)
states[0] = 0
values[0] = 100.0

for t in range(1, N_STEPS):
    states[t] = np.random.choice([0, 1], p=TRANSITION[states[t-1]])
    r = REGIME[states[t]]
    values[t] = values[t-1] + r["mu"] + r["sigma"] * np.random.randn()

times = np.arange(N_STEPS)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 6))
fig.patch.set_facecolor("white")

gs = GridSpec(2, 2, figure=fig,
              left=0.07, right=0.97, top=0.91, bottom=0.09,
              hspace=0.15, wspace=0.32,
              height_ratios=[5, 1],
              width_ratios=[3, 1.4])

ax_main = fig.add_subplot(gs[0, 0])   # time series
ax_ind  = fig.add_subplot(gs[1, 0])   # state strip
ax_mc   = fig.add_subplot(gs[:, 1])   # markov chain — spans both rows

for ax in [ax_main, ax_ind, ax_mc]:
    ax.set_facecolor("#f9f9f9")
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")

# ── Main chart ────────────────────────────────────────────────────────────────
ax_main.set_xlim(0, WINDOW)
ax_main.set_ylim(values.min() - 8, values.max() + 8)
ax_main.set_ylabel("Value", fontsize=10)
ax_main.grid(True, alpha=0.25, color="#dddddd")
ax_main.tick_params(labelbottom=False)

fig.suptitle("Regime-Switching Time Series  |  Markov Chain",
             fontsize=12, fontweight="bold")

state_lbl = ax_main.text(0.98, 0.97, "", transform=ax_main.transAxes,
                          ha="right", va="top", fontsize=10, fontweight="bold")

leg = [mpatches.Patch(color="#22c55e", label="State 0 — Bull ↑"),
       mpatches.Patch(color="#ef4444", label="State 1 — Bear ↓")]
ax_main.legend(handles=leg, fontsize=8, loc="upper left",
               framealpha=0.9, edgecolor="#cccccc")

# ── State indicator strip ─────────────────────────────────────────────────────
ax_ind.set_xlim(0, WINDOW)
ax_ind.set_ylim(0, 1)
ax_ind.set_yticks([])
ax_ind.set_xlabel("Time step", fontsize=9)

# ── Markov chain diagram ──────────────────────────────────────────────────────
ax_mc.axis("off")
ax_mc.set_title("Markov Chain", fontsize=10, pad=6)
ax_mc.set_xlim(0, 1)
ax_mc.set_ylim(0, 1)

# Node positions
POS = {0: (0.25, 0.55), 1: (0.75, 0.55)}
R   = 0.14

# Static arrows (drawn once)
# 0 → 1 (below)
ax_mc.annotate("", xy=(POS[1][0]-R-0.01, POS[1][1]-0.08),
               xytext=(POS[0][0]+R+0.01, POS[0][1]-0.08),
               arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.4,
                               connectionstyle="arc3,rad=-0.35"))
ax_mc.text(0.50, 0.24, f"p₀₁={TRANSITION[0,1]:.2f}",
           ha="center", fontsize=8, color="#999999")

# 1 → 0 (above)
ax_mc.annotate("", xy=(POS[0][0]+R+0.01, POS[0][1]+0.08),
               xytext=(POS[1][0]-R-0.01, POS[1][1]+0.08),
               arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.4,
                               connectionstyle="arc3,rad=-0.35"))
ax_mc.text(0.50, 0.86, f"p₁₀={TRANSITION[1,0]:.2f}",
           ha="center", fontsize=8, color="#999999")

# Self-loop labels
ax_mc.text(POS[0][0], POS[0][1]-R-0.12, f"p₀₀={TRANSITION[0,0]:.2f}",
           ha="center", fontsize=8, color="#22c55e")
ax_mc.text(POS[1][0], POS[1][1]-R-0.12, f"p₁₁={TRANSITION[1,1]:.2f}",
           ha="center", fontsize=8, color="#ef4444")

# μ / σ labels under nodes
ax_mc.text(POS[0][0], 0.05, f"μ={REGIME[0]['mu']:+.1f}  σ={REGIME[0]['sigma']:.1f}",
           ha="center", fontsize=7.5, color="#22c55e")
ax_mc.text(POS[1][0], 0.05, f"μ={REGIME[1]['mu']:+.1f}  σ={REGIME[1]['sigma']:.1f}",
           ha="center", fontsize=7.5, color="#ef4444")

# Node circles — drawn as patches so we can recolour them
node_patches = {}
node_texts   = {}
for s, (cx, cy) in POS.items():
    col = REGIME[s]["color"]
    circ = Circle((cx, cy), R, facecolor="white", edgecolor=col,
                  linewidth=2.5, zorder=3, transform=ax_mc.transAxes,
                  clip_on=False)
    ax_mc.add_patch(circ)
    node_patches[s] = circ
    txt = ax_mc.text(cx, cy + 0.01, str(s), transform=ax_mc.transAxes,
                     ha="center", va="center", fontsize=18, fontweight="bold",
                     color=col, zorder=4)
    node_texts[s] = txt

# ── Animated line segments + indicator bars ───────────────────────────────────
seg_lines = []
ind_bars  = []

for t in range(N_STEPS - 1):
    col = REGIME[states[t + 1]]["color"]
    ln, = ax_main.plot([], [], color=col, linewidth=2.0, alpha=0.0,
                       solid_capstyle="round")
    seg_lines.append(ln)
    bar = mpatches.Rectangle((t, 0), 1, 1, color=col, alpha=0.0)
    ax_ind.add_patch(bar)
    ind_bars.append(bar)


def update(frame):
    t = frame + 1

    # Reveal segments
    for i in range(t - 1):
        seg_lines[i].set_data([times[i], times[i+1]], [values[i], values[i+1]])
        seg_lines[i].set_alpha(1.0)
        ind_bars[i].set_alpha(0.9)

    # Scroll
    x_lo = max(0, t - WINDOW)
    ax_main.set_xlim(x_lo, x_lo + WINDOW)
    ax_ind.set_xlim(x_lo, x_lo + WINDOW)

    # Current state
    cur = states[t - 1]
    r   = REGIME[cur]
    state_lbl.set_text(r["label"])
    state_lbl.set_color(r["color"])

    # Markov chain: highlight active node, dim inactive
    for s, circ in node_patches.items():
        if s == cur:
            circ.set_facecolor(REGIME[s]["color"])
            circ.set_alpha(0.85)
            node_texts[s].set_color("white")
        else:
            circ.set_facecolor("white")
            circ.set_alpha(1.0)
            node_texts[s].set_color(REGIME[s]["color"])

    return seg_lines + ind_bars + [state_lbl] + list(node_patches.values()) + list(node_texts.values())


ani = animation.FuncAnimation(fig, update, frames=N_STEPS - 1,
                               interval=80, blit=False, cache_frame_data=False)

print("Saving GIF …")
ani.save("/mnt/user-data/outputs/regime_switch.gif",
         writer=animation.PillowWriter(fps=15), dpi=110)
print("Done.")