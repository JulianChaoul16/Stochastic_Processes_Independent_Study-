"""
Taxi Arrival — Poisson Process & CTMC
======================================
Run:  python taxi_poisson.py
Keys: +/-  adjust lambda    p  pause    r  reset    q  quit
"""

import curses, time, math, random

LAM_DEFAULT = 2.0
SPEED       = 3.0
CAPACITY    = 12      # dots shown in queue

class Sim:
    def __init__(self, lam=LAM_DEFAULT):
        self.lam     = lam
        self.elapsed = 0.0
        self.count   = 0
        self.history = []
        self.paused  = False
        self.next    = self._exp()

    def _exp(self):
        return self.elapsed + random.expovariate(self.lam)

    def step(self, dt):
        if self.paused: return
        self.elapsed += dt * SPEED
        while self.elapsed >= self.next:
            self.count += 1
            self.history.append(self.next)
            self.next = self._exp()

    def reset(self):
        self.elapsed = 0.0
        self.count   = 0
        self.history = []
        self.next    = self._exp()


def draw(scr, sim):
    scr.erase()
    H, W = scr.getmaxyx()

    YEL  = curses.color_pair(1) | curses.A_BOLD
    CYN  = curses.color_pair(2)
    DIM  = curses.color_pair(3)
    BOLD = curses.A_BOLD

    def put(y, x, s, attr=0):
        if 0 <= y < H and 0 <= x < W:
            try: scr.addstr(y, x, s[:W-x], attr)
            except curses.error: pass

    row = 1

    # Title
    title = "Taxi Arrival Simulator  -  Poisson Process & CTMC"
    put(row, (W - len(title)) // 2, title, BOLD)
    row += 1
    put(row, 2, "-" * (W - 4), DIM)
    row += 2

    # Stats
    status = "PAUSED" if sim.paused else "running"
    stats = (f"  lambda = {sim.lam:.1f} taxis/s   "
             f"arrivals = {sim.count}   "
             f"time = {sim.elapsed:.1f}s   "
             f"E[inter] = {1/sim.lam:.2f}s   "
             f"[{status}]")
    put(row, 0, stats, YEL if sim.paused else CYN)
    row += 2

    # Queue dots
    put(row, 2, "Queue:", BOLD)
    row += 1
    n_dots   = min(sim.count, CAPACITY)
    overflow = sim.count - CAPACITY if sim.count > CAPACITY else 0
    dots     = "  " + "* " * n_dots + ". " * (CAPACITY - n_dots)
    if overflow:
        dots += f"  +{overflow} more"
    put(row, 0, dots)
    for i in range(n_dots):
        try: scr.chgat(row, 2 + i * 2, 1, YEL)
        except curses.error: pass
    row += 2

    # CTMC
    put(row, 2, "CTMC:", BOLD)
    row += 1
    states = ["0", "1", "2", "3", "4+"]
    lstr   = f"lambda={sim.lam:.1f}"
    arrow  = f"--{lstr}-->"
    state  = min(sim.count, 4)
    line   = "  "
    for i, label in enumerate(states):
        line += f"( {label} )"
        if i < len(states) - 1:
            line += arrow
    line += "  ..."
    put(row, 0, line, DIM)
    col = 2
    for i, label in enumerate(states):
        box = f"( {label} )"
        attr = YEL if i == state else CYN if i < state else DIM
        try: scr.chgat(row, col, len(box), attr)
        except curses.error: pass
        col += len(box) + len(arrow)
    row += 2

    # N(t) plot
    put(row, 2, "N(t):", BOLD)
    row += 1
    plot_w = min(W - 10, 58)
    plot_h = 8
    max_t  = max(sim.elapsed, 2.0)
    max_n  = max(sim.count, 5)

    for r in range(plot_h):
        n_val = round(max_n * (plot_h - 1 - r) / (plot_h - 1))
        put(row + r, 2, f"{n_val:2d} |", DIM)

    put(row + plot_h, 4, "   +" + "-" * plot_w, DIM)
    for i in range(5):
        t_val = i / 4 * max_t
        cx    = 7 + int(i / 4 * plot_w)
        put(row + plot_h + 1, cx, f"{t_val:.0f}s", DIM)

    px0 = 7
    prev_cnt = 0
    for col in range(plot_w):
        t   = col / plot_w * max_t
        cnt = sum(1 for h in sim.history if h <= t)
        r   = plot_h - 1 - int(cnt / max_n * (plot_h - 1)) if max_n else plot_h - 1
        r   = max(0, min(plot_h - 1, r))
        if cnt > prev_cnt:
            r_prev = plot_h - 1 - int(prev_cnt / max_n * (plot_h - 1)) if max_n else plot_h - 1
            r_prev = max(0, min(plot_h - 1, r_prev))
            for rr in range(r, r_prev + 1):
                try: scr.addch(row + rr, px0 + col, "|", YEL | curses.A_BOLD)
                except curses.error: pass
        else:
            try: scr.addch(row + r, px0 + col, "-", CYN)
            except curses.error: pass
        prev_cnt = cnt

    row += plot_h + 3

    # Keys
    put(row, 2, "-" * (W - 4), DIM)
    row += 1
    put(row, 2, "+/-  lambda    p  pause    r  reset    q  quit", DIM)

    scr.refresh()


def main(scr):
    curses.curs_set(0)
    scr.nodelay(True)
    scr.timeout(40)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)
    curses.init_pair(2, curses.COLOR_CYAN,   -1)
    curses.init_pair(3, curses.COLOR_WHITE,  -1)

    sim  = Sim()
    last = time.monotonic()

    while True:
        now = time.monotonic()
        sim.step(now - last)
        last = now
        draw(scr, sim)
        ch = scr.getch()
        if   ch in (ord('q'), ord('Q')): break
        elif ch in (ord('p'), ord('P')): sim.paused = not sim.paused
        elif ch in (ord('r'), ord('R')): sim.reset()
        elif ch == ord('+'): sim.lam = min(10.0, round(sim.lam + 0.5, 1))
        elif ch == ord('-'): sim.lam = max(0.5,  round(sim.lam - 0.5, 1))


curses.wrapper(main)