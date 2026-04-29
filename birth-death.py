"""
Birth-Death Process — Matplotlib Animation
==========================================
Simulates a continuous-time birth-death Markov chain and animates:
  • Population dots (births = green появляются, deaths = red исчезают)
  • N(t) trajectory over time
  • CTMC state diagram with current state highlighted

Usage:
    python birth_death.py
    python birth_death.py --birth 1.5 --death 1.0 --cap 10
"""

import argparse
import random
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec

# ── Parameters ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--birth", type=float, default=1.2, help="Birth rate λ (default 1.2)")
parser.add_argument("--death", type=float, default=0.8, help="Death rate μ (default 0.8)")
parser.add_argument("--cap",   type=int,   default=8,   help="Max population / CTMC states shown (default 8)")
parser.add_argument("--speed", type=float, default=1.5, help="Sim speed multiplier (default 1.5)")
args = parser.parse_args()

LAM   = args.birth
MU    = args.death
CAP   = args.cap
SPEED = args.speed

HISTORY_WINDOW = 20.0   # seconds of N(t) shown in trajectory plot
MAX_DOTS       = CAP    # dots on screen

# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#0f0f14"
PANEL_BG  = "#16161e"
GRID_COL  = "#2a2a38"
TEXT_COL  = "#cccbe0"
BIRTH_COL = "#4ade80"   # green
DEATH_COL = "#f87171"   # red
TRAJ_COL  = "#818cf8"   # indigo
STATE_DEF = "#2a2a4a"   # default state circle
STATE_ACT = "#facc15"   # active state (yellow)
STATE_VIS = "#6366f1"   # visited (purple)
ARROW_COL = "#64748b"

matplotlib.rcParams.update({
    "figure.facecolor": BG,
    "text.color":       TEXT_COL,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   GRID_COL,
    "axes.labelcolor":  TEXT_COL,
    "xtick.color":      TEXT_COL,
    "ytick.color":      TEXT_COL,
    "grid.color":       GRID_COL,
    "font.family":      "monospace",
})

# ── Simulation ────────────────────────────────────────────────────────────────

class BirthDeath:
    def __init__(self):
        self.t        = 0.0
        self.n        = 3          # start with small population
        self.t_hist   = [0.0]
        self.n_hist   = [3]
        self.last_event = None     # "birth" | "death" | None
        self._schedule()

    def _exp(self, rate):
        return random.expovariate(rate) if rate > 0 else float("inf")

    def _schedule(self):
        birth_rate = LAM * self.n if self.n > 0 else LAM   # linear birth
        death_rate = MU  * self.n                           # linear death
        self._next_birth = self.t + self._exp(birth_rate)
        self._next_death = self.t + self._exp(death_rate) if self.n > 0 else float("inf")

    def step(self, dt):
        self.t += dt * SPEED
        events = 0
        while True:
            if self._next_birth <= self._next_death and self._next_birth <= self.t:
                self.t_hist.append(self._next_birth)
                self.n += 1
                self.n_hist.append(self.n)
                self.last_event = "birth"
                self._schedule()
                events += 1
            elif self._next_death < self._next_birth and self._next_death <= self.t and self.n > 0:
                self.t_hist.append(self._next_death)
                self.n -= 1
                self.n_hist.append(self.n)
                self.last_event = "death"
                self._schedule()
                events += 1
            else:
                break
            if events > 20:
                break
        return events > 0

sim = BirthDeath()

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 8), facecolor=BG)
fig.canvas.manager.set_window_title("Birth-Death Process — CTMC Simulation")

gs = GridSpec(3, 2, figure=fig,
              left=0.07, right=0.97, top=0.92, bottom=0.08,
              hspace=0.55, wspace=0.35,
              height_ratios=[1.1, 1.4, 1.0])

ax_pop   = fig.add_subplot(gs[0, :])   # population dots — full width
ax_traj  = fig.add_subplot(gs[1, 0])   # N(t) trajectory
ax_ctmc  = fig.add_subplot(gs[1, 1])   # CTMC state diagram
ax_info  = fig.add_subplot(gs[2, :])   # stats / legend bar

for ax in [ax_pop, ax_traj, ax_ctmc, ax_info]:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)

# ── Population panel ──────────────────────────────────────────────────────────
ax_pop.set_xlim(0, MAX_DOTS + 1)
ax_pop.set_ylim(-0.5, 1.5)
ax_pop.set_yticks([])
ax_pop.set_xticks([])
ax_pop.set_title("Population", color=TEXT_COL, fontsize=11, pad=6, loc="left")

pop_dots = []
for i in range(MAX_DOTS):
    dot, = ax_pop.plot([], [], "o", markersize=18, color=STATE_DEF, alpha=0.3,
                       markeredgecolor="none")
    pop_dots.append(dot)

flash_dot, = ax_pop.plot([], [], "o", markersize=26, color=BIRTH_COL,
                          alpha=0.0, markeredgecolor="none", zorder=5)

pop_label = ax_pop.text(0.5, -0.3, "", transform=ax_pop.transAxes,
                         ha="center", color=TEXT_COL, fontsize=10)

# ── Trajectory panel ──────────────────────────────────────────────────────────
ax_traj.set_title("N(t) — population over time", color=TEXT_COL,
                   fontsize=11, pad=6, loc="left")
ax_traj.set_xlabel("time (s)", fontsize=9)
ax_traj.set_ylabel("N(t)", fontsize=9)
ax_traj.grid(True, alpha=0.3)
ax_traj.set_ylim(-0.5, CAP + 1.5)

traj_line, = ax_traj.step([], [], where="post", color=TRAJ_COL,
                            linewidth=2, zorder=3)
traj_dot,  = ax_traj.plot([], [], "o", color=STATE_ACT, markersize=7, zorder=4)

birth_scatter = ax_traj.scatter([], [], color=BIRTH_COL, s=50, zorder=5,
                                 marker="^", label="birth")
death_scatter = ax_traj.scatter([], [], color=DEATH_COL, s=50, zorder=5,
                                 marker="v", label="death")
ax_traj.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COL, loc="upper left")

# ── CTMC diagram ──────────────────────────────────────────────────────────────
ax_ctmc.set_xlim(-0.5, CAP + 0.5)
ax_ctmc.set_ylim(-1.2, 1.8)
ax_ctmc.set_aspect("equal")
ax_ctmc.axis("off")
ax_ctmc.set_title("CTMC state diagram", color=TEXT_COL, fontsize=11, pad=6, loc="left")

STATE_R = 0.35
STATES  = list(range(min(CAP + 1, 9)))   # 0 … CAP (max 9 for spacing)
spacing = (CAP) / max(len(STATES) - 1, 1)

state_circles = []
state_labels  = []
birth_arrows  = []
death_arrows  = []

for i, s in enumerate(STATES):
    cx = i * spacing
    circ = Circle((cx, 0), STATE_R, color=STATE_DEF,
                   ec=GRID_COL, linewidth=1.2, zorder=3)
    ax_ctmc.add_patch(circ)
    state_circles.append(circ)

    lbl = ax_ctmc.text(cx, 0, str(s), ha="center", va="center",
                        fontsize=9 if len(STATES) <= 7 else 7,
                        color=TEXT_COL, fontweight="bold", zorder=4)
    state_labels.append(lbl)

    if i < len(STATES) - 1:
        nx = (i + 1) * spacing
        # birth arrow (below, left→right)
        ax_ctmc.annotate("", xy=(nx - STATE_R - 0.02, -0.18),
                          xytext=(cx + STATE_R + 0.02, -0.18),
                          arrowprops=dict(arrowstyle="->", color=BIRTH_COL,
                                          lw=1.4, connectionstyle="arc3,rad=-0.25"))
        bl = ax_ctmc.text((cx + nx) / 2, -0.85,
                           f"λ·n", ha="center", va="center",
                           fontsize=7, color=BIRTH_COL)
        birth_arrows.append(bl)

        # death arrow (above, right→left)
        ax_ctmc.annotate("", xy=(cx + STATE_R + 0.02, 0.18),
                          xytext=(nx - STATE_R - 0.02, 0.18),
                          arrowprops=dict(arrowstyle="->", color=DEATH_COL,
                                          lw=1.4, connectionstyle="arc3,rad=-0.25"))
        dl = ax_ctmc.text((cx + nx) / 2, 0.85,
                           f"μ·n", ha="center", va="center",
                           fontsize=7, color=DEATH_COL)
        death_arrows.append(dl)

ctmc_cursor, = ax_ctmc.plot([], [], "o", markersize=28,
                              color=STATE_ACT, alpha=0.25, zorder=2)

# ── Info bar ──────────────────────────────────────────────────────────────────
ax_info.axis("off")
info_text = ax_info.text(0.5, 0.5, "", transform=ax_info.transAxes,
                          ha="center", va="center", fontsize=10,
                          color=TEXT_COL, family="monospace")

# Suptitle
fig.suptitle("Birth-Death Process  |  CTMC Simulation",
             color=TEXT_COL, fontsize=13, fontweight="bold", y=0.97)

# ── Flash state for recent event ──────────────────────────────────────────────
flash_alpha  = 0.0
flash_state  = -1
flash_color  = BIRTH_COL

# ── Update function ───────────────────────────────────────────────────────────

def update(frame):
    global flash_alpha, flash_state, flash_color

    dt = 0.04  # ~25 fps
    had_event = sim.step(dt)

    n = sim.n
    t = sim.t

    # ── Population dots ────────────────────────────────────────────────────
    for i, dot in enumerate(pop_dots):
        if i < n:
            dot.set_data([i + 1], [0.5])
            dot.set_color(BIRTH_COL if i == n - 1 and sim.last_event == "birth"
                          else "#818cf8")
            dot.set_alpha(1.0)
            dot.set_markersize(18)
        else:
            dot.set_data([], [])

    pop_label.set_text(f"N = {n}  |  λ={LAM:.1f}  μ={MU:.1f}")

    # Flash effect on last dot
    if had_event:
        flash_state = n if sim.last_event == "birth" else n
        flash_color = BIRTH_COL if sim.last_event == "birth" else DEATH_COL
        flash_alpha = 0.85

    flash_alpha = max(0.0, flash_alpha - 0.06)
    if flash_alpha > 0 and 0 < n <= MAX_DOTS:
        pos = n if sim.last_event == "birth" else n + 1
        pos = max(1, min(pos, MAX_DOTS))
        flash_dot.set_data([pos], [0.5])
        flash_dot.set_color(flash_color)
        flash_dot.set_alpha(flash_alpha)
    else:
        flash_dot.set_data([], [])

    # ── Trajectory ─────────────────────────────────────────────────────────
    t_hist = np.array(sim.t_hist)
    n_hist = np.array(sim.n_hist)

    t_min = max(0.0, t - HISTORY_WINDOW)
    ax_traj.set_xlim(t_min, t_min + HISTORY_WINDOW)

    mask = t_hist >= t_min
    if mask.any():
        traj_line.set_data(t_hist[mask], n_hist[mask])
    traj_dot.set_data([t], [n])

    # birth/death markers
    births_x, births_y = [], []
    deaths_x, deaths_y = [], []
    if len(sim.n_hist) > 1:
        for i in range(1, len(sim.n_hist)):
            if t_hist[i] < t_min:
                continue
            dn = sim.n_hist[i] - sim.n_hist[i-1]
            if dn > 0:
                births_x.append(t_hist[i]); births_y.append(sim.n_hist[i])
            elif dn < 0:
                deaths_x.append(t_hist[i]); deaths_y.append(sim.n_hist[i])

    birth_scatter.set_offsets(np.c_[births_x, births_y] if births_x
                               else np.empty((0, 2)))
    death_scatter.set_offsets(np.c_[deaths_x, deaths_y] if deaths_x
                               else np.empty((0, 2)))

    # ── CTMC ───────────────────────────────────────────────────────────────
    for i, circ in enumerate(state_circles):
        s = STATES[i]
        if s == n:
            circ.set_facecolor(STATE_ACT)
            circ.set_edgecolor(STATE_ACT)
            circ.set_linewidth(2.5)
        elif s < n:
            circ.set_facecolor(STATE_VIS)
            circ.set_edgecolor(STATE_VIS)
            circ.set_linewidth(1.2)
        else:
            circ.set_facecolor(STATE_DEF)
            circ.set_edgecolor(GRID_COL)
            circ.set_linewidth(1.2)

        state_labels[i].set_color("#0f0f14" if s == n else TEXT_COL)

    if n < len(STATES):
        cx = STATES.index(n) * spacing
        ctmc_cursor.set_data([cx], [0])
    else:
        ctmc_cursor.set_data([], [])

    # ── Info bar ───────────────────────────────────────────────────────────
    last_ev = sim.last_event or "—"
    ev_col  = BIRTH_COL if last_ev == "birth" else DEATH_COL if last_ev == "death" else TEXT_COL
    births_total = sum(1 for i in range(1, len(sim.n_hist)) if sim.n_hist[i] > sim.n_hist[i-1])
    deaths_total = sum(1 for i in range(1, len(sim.n_hist)) if sim.n_hist[i] < sim.n_hist[i-1])

    info = (f"t = {t:6.2f}s   "
            f"N(t) = {n:3d}   "
            f"births = {births_total:4d}   "
            f"deaths = {deaths_total:4d}   "
            f"last event: {last_ev.upper()}")
    info_text.set_text(info)
    info_text.set_color(ev_col if had_event else TEXT_COL)

    return (traj_line, traj_dot, flash_dot, birth_scatter, death_scatter,
            ctmc_cursor, info_text, pop_label, *pop_dots, *state_circles, *state_labels)


ani = animation.FuncAnimation(fig, update, interval=40,
                               blit=False, cache_frame_data=False)

plt.show() 