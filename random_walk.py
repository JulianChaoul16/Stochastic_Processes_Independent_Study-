import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def save_as_gif(n_steps=200, filename="random_walk.gif"):
    np.random.seed(42)

    # Pre-generate all moves
    moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    choices = np.random.randint(0, 4, size=n_steps)
    steps = moves[choices]

    # Calculate all positions
    positions = np.zeros((n_steps + 1, 2))
    positions[1:] = np.cumsum(steps, axis=0)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)

    # Initialize plot elements
    line, = ax.plot([], [], 'b-', alpha=0.7, linewidth=2)
    point, = ax.plot([], [], 'ro', markersize=8)
    start_point, = ax.plot([0], [0], 'gs', markersize=10, label='Start')

    # Set up the plot
    margin = max(np.max(np.abs(positions)) * 1.1, 10)
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.grid(True, alpha=0.3)
    ax.set_title('2D Random Walk', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    # Add step counter
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

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_steps + 1,
                                   interval=150, blit=True, repeat=True)

    # Save as GIF
    print(f"Saving animation as {filename}...")
    try:
        anim.save(filename, writer='pillow', fps=8)
        print(f"GIF saved successfully as {filename}!")
    except Exception as e:
        print(f"Error saving GIF: {e}")

    plt.close()
    return filename


def animated_random_walk_2d(n_steps=500, interval=50, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Pre-generate all moves for smoother animation
    moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    choices = np.random.randint(0, 4, size=n_steps)
    steps = moves[choices]

    # Calculate all positions
    positions = np.zeros((n_steps + 1, 2))
    positions[1:] = np.cumsum(steps, axis=0)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Initialize empty line objects
    line, = ax.plot([], [], 'b-', alpha=0.7, linewidth=1.5)
    point, = ax.plot([], [], 'ro', markersize=8)
    start_point, = ax.plot([0], [0], 'gs', markersize=10, label='Start')

    # Set up the plot
    margin = max(np.max(np.abs(positions)) * 1.1, 10)
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.grid(True, alpha=0.3)
    ax.set_title('2D Random Walk Animation')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    # Add step counter text
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

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_steps + 1,
                                   interval=interval, blit=True, repeat=True)

    plt.show()
    return anim, positions


# Save as GIF
gif_file = save_as_gif(n_steps=150, filename="random_walk.gif")

# Run the interactive animation
anim, positions = animated_random_walk_2d(n_steps=300, interval=30, seed=42)