"""
SIR Network Animation – COVID-19 Spread
========================================
White background, neutral muted palette, large prominent network.
Each node = 100 people. Edges show transmission pathways.

Colors:
  Steel blue  = Susceptible
  Warm amber  = Infected
  Sage green  = Recovered
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# ──────────────────────────────────────────────
#  PARAMETERS
# ──────────────────────────────────────────────
PEOPLE_PER_NODE  = 100
N_NODES          = 100
BETA_MEAN        = 0.30
BETA_STD         = 0.05
GAMMA            = 0.10
EDGE_PROB        = 0.08
SEED_NODES       = 2
T_DAYS           = 80
OUTPUT_DIR = r"C:\Users\JuJuC\OneDrive\Desktop\IS\programs"# ← change for your machine

FPS              = 6
DPI              = 120

# ──────────────────────────────────────────────
#  NEUTRAL PALETTE  (works on white bg)
# ──────────────────────────────────────────────
COLOR_S  = np.array([0.35, 0.55, 0.75])   # steel blue
COLOR_I  = np.array([0.82, 0.50, 0.22])   # warm amber/burnt orange
COLOR_R  = np.array([0.40, 0.65, 0.48])   # sage green

HEX_S    = "#5a8dbf"
HEX_I    = "#d08038"
HEX_R    = "#67a67b"

BG       = "white"
EDGE_COL = "#c0c8d4"   # light blue-grey edges (resting)
EDGE_ACT = "#d08038"   # amber edge when transmitting
TEXT_COL = "#2a2f3a"   # near-black text

# ──────────────────────────────────────────────
#  BUILD NETWORK
# ──────────────────────────────────────────────
rng = np.random.default_rng(seed=7)

angles = np.linspace(0, 2 * np.pi, N_NODES, endpoint=False)
radius = np.ones(N_NODES) + rng.uniform(-0.30, 0.30, N_NODES)
pos_x  = radius * np.cos(angles) + rng.uniform(-0.12, 0.12, N_NODES)
pos_y  = radius * np.sin(angles) + rng.uniform(-0.12, 0.12, N_NODES)

adj = np.zeros((N_NODES, N_NODES), dtype=bool)
for i in range(N_NODES):
    adj[i, (i+1) % N_NODES] = True
    adj[(i+1) % N_NODES, i] = True
    for j in range(i+2, N_NODES):
        if rng.random() < EDGE_PROB:
            adj[i, j] = True
            adj[j, i] = True

edges = [(i, j) for i in range(N_NODES)
         for j in range(i+1, N_NODES) if adj[i, j]]

# ──────────────────────────────────────────────
#  NODE-LEVEL SIR
# ──────────────────────────────────────────────
S = np.ones(N_NODES)
I = np.zeros(N_NODES)
R = np.zeros(N_NODES)

for idx in rng.choice(N_NODES, size=SEED_NODES, replace=False):
    I[idx] = 1.0
    S[idx] = 0.0

def step(S, I, R, adj, beta_mean, beta_std, gamma, rng):
    S_new = S.copy(); I_new = I.copy(); R_new = R.copy()
    beta  = max(0.0, rng.normal(beta_mean, beta_std))
    for node in range(len(S)):
        local_force = beta * S[node] * I[node]
        neighbours  = np.where(adj[node])[0]
        cross_force = (beta * 0.3 * S[node] * I[neighbours].mean()
                       if len(neighbours) > 0 else 0.0)
        new_inf = min(S[node], local_force + cross_force)
        new_rec = gamma * I[node]
        S_new[node] = max(0.0, S[node] - new_inf)
        I_new[node] = max(0.0, I[node] + new_inf - new_rec)
        R_new[node] = min(1.0, R[node] + new_rec)
    return S_new, I_new, R_new

# ── Run & store ──
print("Running network simulation...")
history = [(S.copy(), I.copy(), R.copy())]
for _ in range(T_DAYS):
    S, I, R = step(S, I, R, adj, BETA_MEAN, BETA_STD, GAMMA, rng)
    history.append((S.copy(), I.copy(), R.copy()))
print(f"  {T_DAYS} days simulated, {len(history)} frames")

# ──────────────────────────────────────────────
#  FIGURE  – wider, network dominates left 2/3
# ──────────────────────────────────────────────
def node_color(s, i, r):
    return s * COLOR_S + i * COLOR_I + r * COLOR_R

fig = plt.figure(figsize=(15, 7.5), facecolor=BG)

# Network takes up 68% of width
ax_net = fig.add_axes([0.01, 0.06, 0.60, 0.88])   # [left, bottom, w, h]
ax_bar = fig.add_axes([0.68, 0.12, 0.28, 0.72])

ax_net.set_facecolor(BG)
ax_net.set_aspect("equal")
ax_net.axis("off")

ax_bar.set_facecolor(BG)

# ── Draw edges ──
edge_lines = []
for (i, j) in edges:
    ln, = ax_net.plot(
        [pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]],
        color=EDGE_COL, alpha=0.55, linewidth=1.0, zorder=1,
        solid_capstyle="round"
    )
    edge_lines.append(ln)

# ── Draw nodes ──
init_colors = [node_color(history[0][0][n], history[0][1][n], history[0][2][n])
               for n in range(N_NODES)]

# Outer halo (translucent, slightly larger)
halo = ax_net.scatter(
    pos_x, pos_y,
    c=init_colors, s=520, alpha=0.22, zorder=2, edgecolors="none"
)

# Main node
scatter = ax_net.scatter(
    pos_x, pos_y,
    c=init_colors, s=200, zorder=3,
    edgecolors="white", linewidths=1.5
)

# ── Day label & title ──
day_text = ax_net.text(
    0.02, 0.98, "Day 0",
    transform=ax_net.transAxes,
    fontsize=15, color=TEXT_COL, fontweight="bold", va="top",
    fontfamily="monospace"
)

ax_net.set_title(
    "COVID-19 Network Spread  ·  Each node = 100 people",
    color=TEXT_COL, fontsize=13, fontweight="semibold", pad=6
)

# ── Legend inside network panel ──
legend_handles = [
    mpatches.Patch(color=HEX_S, label="Susceptible"),
    mpatches.Patch(color=HEX_I, label="Infected"),
    mpatches.Patch(color=HEX_R, label="Recovered"),
]
ax_net.legend(
    handles=legend_handles, loc="lower left",
    framealpha=0.85, frameon=True, edgecolor="#dde2e8",
    facecolor="white", labelcolor=TEXT_COL, fontsize=10,
    handlelength=1.2, handleheight=1.0
)

# ──────────────────────────────────────────────
#  BAR CHART
# ──────────────────────────────────────────────
total_pop = N_NODES * PEOPLE_PER_NODE
categories = ["Susceptible", "Infected", "Recovered"]
bar_colors = [HEX_S, HEX_I, HEX_R]

init_counts = [
    history[0][0].sum() * PEOPLE_PER_NODE,
    history[0][1].sum() * PEOPLE_PER_NODE,
    history[0][2].sum() * PEOPLE_PER_NODE,
]

bars = ax_bar.bar(
    categories, init_counts,
    color=bar_colors, edgecolor="white", linewidth=1.2,
    width=0.55
)

ax_bar.set_ylim(0, total_pop * 1.12)
ax_bar.set_ylabel("Number of people", color=TEXT_COL, fontsize=10)
ax_bar.set_title("Population Status", color=TEXT_COL, fontsize=12,
                 fontweight="semibold", pad=8)

for spine in ["top", "right"]:
    ax_bar.spines[spine].set_visible(False)
ax_bar.spines["left"].set_color("#c8cdd6")
ax_bar.spines["bottom"].set_color("#c8cdd6")
ax_bar.tick_params(colors=TEXT_COL, length=3)
for lbl in ax_bar.get_xticklabels():
    lbl.set_color(TEXT_COL); lbl.set_fontsize(9)
for lbl in ax_bar.get_yticklabels():
    lbl.set_color(TEXT_COL); lbl.set_fontsize(9)
ax_bar.yaxis.label.set_color(TEXT_COL)

bar_labels = [
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total_pop * 0.015,
        f"{int(v):,}", ha="center", va="bottom",
        color=TEXT_COL, fontsize=9, fontweight="bold"
    )
    for bar, v in zip(bars, init_counts)
]

# Horizontal gridlines (subtle)
ax_bar.yaxis.grid(True, color="#e8eaed", linewidth=0.8, zorder=0)
ax_bar.set_axisbelow(True)

# ──────────────────────────────────────────────
#  ANIMATION UPDATE
# ──────────────────────────────────────────────
def update(frame):
    S_f, I_f, R_f = history[frame]

    colors = [node_color(S_f[n], I_f[n], R_f[n]) for n in range(N_NODES)]
    scatter.set_facecolor(colors)
    halo.set_facecolor(colors)

    # Size: infected nodes grow, recovered nodes stay medium
    sizes_main = 160 + I_f * 260 + R_f * 20
    sizes_halo = 480 + I_f * 700
    scatter.set_sizes(sizes_main)
    halo.set_sizes(sizes_halo)

    # Update edges: amber + thicker when both endpoints infected
    for (i, j), ln in zip(edges, edge_lines):
        activity = (I_f[i] + I_f[j]) / 2
        if activity > 0.08:
            ln.set_color(EDGE_ACT)
            ln.set_alpha(min(0.85, 0.2 + activity * 1.2))
            ln.set_linewidth(min(3.5, 0.8 + activity * 4.0))
        else:
            ln.set_color(EDGE_COL)
            ln.set_alpha(0.45)
            ln.set_linewidth(0.9)

    day_text.set_text(f"Day {frame}")

    # Update bars
    counts = [
        S_f.sum() * PEOPLE_PER_NODE,
        I_f.sum() * PEOPLE_PER_NODE,
        R_f.sum() * PEOPLE_PER_NODE,
    ]
    for bar, val, lbl in zip(bars, counts, bar_labels):
        bar.set_height(val)
        lbl.set_y(val + total_pop * 0.015)
        lbl.set_text(f"{int(val):,}")

    return [scatter, halo, day_text] + list(bars) + bar_labels + edge_lines


anim = FuncAnimation(fig, update, frames=len(history),
                     interval=1000 // FPS, blit=False)

out_path = os.path.join(OUTPUT_DIR, "sir_network.gif")
print(f"Saving GIF → {out_path} ...")
anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
plt.close()
print("✓ Done!")