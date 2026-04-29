import numpy as np
import matplotlib.pyplot as plt

# ── Simulate ──────────────────────────────────────────────────────────────────
np.random.seed(42)
steps     = 100
times     = np.linspace(0, 100, steps)
dt        = times[1] - times[0]
returns   = 0.05 * dt + 0.15 * np.sqrt(dt) * np.random.randn(steps)
prices    = np.empty(steps)
prices[0] = 100.0
for i in range(1, steps):
    prices[i] = prices[i-1] * np.exp(returns[i])

price_at = {i: round(prices[i], 2) for i in range(steps)}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle("Makebelieve Corp (MKBL)", fontsize=13)

# Stock chart
ax1.plot(times, prices, color="black", linewidth=1.5)
ax1.set_ylabel("Price ($)")
ax1.grid(True, alpha=0.3)

# Array grid
ax2.set_title("price_at[ ] — index → price", fontsize=10, loc="left")
ax2.axis("off")

COLS = 20
for idx in range(steps):
    row = idx // COLS
    col = idx  % COLS
    x   = col / COLS
    y   = 1.0 - (row + 1) / (steps // COLS)

    ax2.add_patch(plt.Rectangle((x, y), 1/COLS, 1/(steps//COLS),
                                 facecolor="white", edgecolor="gray",
                                 linewidth=0.5, transform=ax2.transAxes))
    ax2.text(x + 0.5/COLS, y + 0.65/(steps//COLS), str(idx),
             transform=ax2.transAxes, ha="center", fontsize=5.5, color="gray")
    ax2.text(x + 0.5/COLS, y + 0.2/(steps//COLS), f"${prices[idx]:.1f}",
             transform=ax2.transAxes, ha="center", fontsize=6, color="black")

plt.tight_layout()
plt.savefig("stock.png", dpi=150)   # saves next to the script
plt.show()