import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
np.random.seed(0)
 
N = 60
W, H = 10, 10
FRAMES = 80
FPS = 12
 
COLORS = (
    ['#e84040'] * N +
    ['#44cc66'] * N +
    ['#4488ff'] * N
)
 
 
def make_low_entropy_positions(n, width=W, height=H):
    positions = np.zeros((3 * n, 2))
    centers = [
        (width * 0.20, height * 0.50),
        (width * 0.50, height * 0.78),
        (width * 0.80, height * 0.50),
    ]
    for i, (cx, cy) in enumerate(centers):
        pts = np.random.randn(n, 2) * 0.55 + [cx, cy]
        pts[:, 0] = np.clip(pts[:, 0], 0.6, width - 0.6)
        pts[:, 1] = np.clip(pts[:, 1], 0.6, height - 0.6)
        positions[i * n:(i + 1) * n] = pts
    return positions
 
 
def make_high_entropy_positions(n, width=W, height=H):
    return np.random.uniform([0.6, 0.6], [width - 0.6, height - 0.6],
                             size=(3 * n, 2))
 
 
def update_positions(positions, velocities, width=W, height=H, speed=0.12):
    velocities += np.random.randn(*velocities.shape) * speed * 0.05
    velocities *= 0.92
    positions += velocities
    for axis, limit in [(0, width), (1, height)]:
        lo = positions[:, axis] < 0.6
        hi = positions[:, axis] > limit - 0.6
        positions[lo, axis] = 0.6
        positions[hi, axis] = limit - 0.6
        velocities[lo, axis] *= -1
        velocities[hi, axis] *= -1
    return positions, velocities
 
 
def grid_entropy(positions, n_cells=8, width=W, height=H):
    counts, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=n_cells,
        range=[[0, width], [0, height]]
    )
    p = counts.ravel() / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))
 
 
# ── LOW ENTROPY GIF ────────────────────────────────────────────────────────────
print("Rendering low entropy GIF...")
 
np.random.seed(0)
pos_lo = make_low_entropy_positions(N)
vel_lo = np.random.randn(3 * N, 2) * 0.05
 
fig_lo, ax_lo = plt.subplots(figsize=(6, 6), facecolor='#0d0d0d')
ax_lo.set_facecolor('#111111')
ax_lo.set_xlim(0, W); ax_lo.set_ylim(0, H)
ax_lo.set_aspect('equal')
ax_lo.set_xticks([]); ax_lo.set_yticks([])
for spine in ax_lo.spines.values():
    spine.set_edgecolor('#333333')
ax_lo.set_title("Low Entropy  —  Ordered", color='white', fontsize=13,
                 fontweight='bold', pad=10)
 
scat_lo = ax_lo.scatter(pos_lo[:, 0], pos_lo[:, 1],
                         c=COLORS, s=40, alpha=0.90, linewidths=0)
 
ent_lo_text = ax_lo.text(0.03, 0.96, '', transform=ax_lo.transAxes,
                          color='#cccccc', fontsize=10, va='top', ha='left',
                          bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='#222222', edgecolor='#444444', alpha=0.85))
plt.tight_layout()
 
 
def animate_lo(frame):
    global pos_lo, vel_lo
    pos_lo, vel_lo = update_positions(pos_lo, vel_lo, speed=0.04)
    scat_lo.set_offsets(pos_lo)
    ent_lo_text.set_text(f"S ≈ {grid_entropy(pos_lo):.2f} nats")
    return scat_lo, ent_lo_text
 
 
anim_lo = animation.FuncAnimation(fig_lo, animate_lo, frames=FRAMES,
                                   interval=1000 // FPS, blit=True)
anim_lo.save("/home/claude/low_entropy.gif", writer='pillow', fps=FPS)
plt.close(fig_lo)
print("  low_entropy.gif saved.")
 
 
# ── HIGH ENTROPY GIF ───────────────────────────────────────────────────────────
print("Rendering high entropy GIF...")
 
np.random.seed(1)
pos_hi = make_high_entropy_positions(N)
vel_hi = np.random.randn(3 * N, 2) * 0.15
 
fig_hi, ax_hi = plt.subplots(figsize=(6, 6), facecolor='#0d0d0d')
ax_hi.set_facecolor('#111111')
ax_hi.set_xlim(0, W); ax_hi.set_ylim(0, H)
ax_hi.set_aspect('equal')
ax_hi.set_xticks([]); ax_hi.set_yticks([])
for spine in ax_hi.spines.values():
    spine.set_edgecolor('#333333')
ax_hi.set_title("High Entropy  —  Disordered", color='white', fontsize=13,
                 fontweight='bold', pad=10)
 
scat_hi = ax_hi.scatter(pos_hi[:, 0], pos_hi[:, 1],
                         c=COLORS, s=40, alpha=0.90, linewidths=0)
 
ent_hi_text = ax_hi.text(0.03, 0.96, '', transform=ax_hi.transAxes,
                          color='#cccccc', fontsize=10, va='top', ha='left',
                          bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='#222222', edgecolor='#444444', alpha=0.85))
plt.tight_layout()
 
 
def animate_hi(frame):
    global pos_hi, vel_hi
    pos_hi, vel_hi = update_positions(pos_hi, vel_hi, speed=0.18)
    scat_hi.set_offsets(pos_hi)
    ent_hi_text.set_text(f"S ≈ {grid_entropy(pos_hi):.2f} nats")
    return scat_hi, ent_hi_text
 
 
anim_hi = animation.FuncAnimation(fig_hi, animate_hi, frames=FRAMES,
                                   interval=1000 // FPS, blit=True)
anim_hi.save("/home/claude/high_entropy.gif", writer='pillow', fps=FPS)
plt.close(fig_hi)
print("  high_entropy.gif saved.")
 
print("Done.")