import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def save_as_gif(n_steps=200, filename="random_walk.gif", drift=0.4):
    np.random.seed(42)

    moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    
    # Weighted probabilities: [right, left, up, down]
    other = (1 - drift) / 3
    probs = [drift, other, other, other]
    choices = np.random.choice(4, size=n_steps, p=probs)
    steps = moves[choices]

    positions = np.zeros((n_steps + 1, 2))
    positions[1:] = np.cumsum(steps, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    line, = ax.plot([], [], 'b-', alpha=0.7, linewidth=2)
    point, = ax.plot([], [], 'ro', markersize=8)
    start_point, = ax.plot([0], [0], 'gs', markersize=10, label='Start')

    margin = max(np.max(np.abs(positions)) * 1.1, 10)
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'2D Random Walk (drift={drift:.0%} right)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    def animate(frame):
        x_data = positions[:frame + 1, 0]
        y_data = positions[:frame + 1, 1]
        line.set_data(x_data, y_data)
        if frame > 0:
            point.set_data([x_data[-1]], [y_data[-1]])
        else:
            point.set_data([], [])
        step_text.set_text(f'Step: {frame}')
        return line, point, step_text

    anim = animation.FuncAnimation(fig, animate, frames=n_steps + 1,
                                   interval=150, blit=True, repeat=True)
    anim.save(filename, writer='pillow', fps=8)
    plt.close()
    return filename


def animated_random_walk_2d(n_steps=500, interval=50, seed=None, drift=0.4):
    if seed is not None:
        np.random.seed(seed)

    moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    # Weighted probabilities: [right, left, up, down]
    other = (1 - drift) / 3
    probs = [drift, other, other, other]
    choices = np.random.choice(4, size=n_steps, p=probs)
    steps = moves[choices]

    positions = np.zeros((n_steps + 1, 2))
    positions[1:] = np.cumsum(steps, axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    line, = ax.plot([], [], 'b-', alpha=0.7, linewidth=1.5)
    point, = ax.plot([], [], 'ro', markersize=8)
    start_point, = ax.plot([0], [0], 'gs', markersize=10, label='Start')

    margin = max(np.max(np.abs(positions)) * 1.1, 10)
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'2D Random Walk Animation (drift={drift:.0%} right)')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def animate(frame):
        x_data = positions[:frame + 1, 0]
        y_data = positions[:frame + 1, 1]
        line.set_data(x_data, y_data)
        if frame > 0:
            point.set_data([x_data[-1]], [y_data[-1]])
        else:
            point.set_data([], [])
        step_text.set_text(f'Step: {frame}/{n_steps}')
        return line, point, step_text

    anim = animation.FuncAnimation(fig, animate, frames=n_steps + 1,
                                   interval=interval, blit=True, repeat=True)
    plt.show()
    return anim, positions


gif_file = save_as_gif(n_steps=150, filename="random_walk_drift.gif", drift=0.4)
anim, positions = animated_random_walk_2d(n_steps=300, interval=30, seed=42, drift=0.4)