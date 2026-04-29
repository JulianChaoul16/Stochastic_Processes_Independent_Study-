"""
Markov Chain Random Walk
========================
4 states representing compass directions (N, S, E, W).
From any state, there is a 1/4 probability of transitioning
to each of the 4 states (including staying in the same direction).
"""

import random
import sys
from collections import Counter


# ---------------------------------------------------------------------------
# Transition matrix
# Each state maps to 4 possible next states, each with probability 0.25
# ---------------------------------------------------------------------------

STATES = ["N", "S", "E", "W"]

TRANSITION_MATRIX: dict[str, dict[str, float]] = {
    state: {next_state: 0.25 for next_state in STATES}
    for state in STATES
}


def next_state(current: str) -> str:
    """Return the next state by sampling from the transition distribution."""
    transitions = TRANSITION_MATRIX[current]
    states = list(transitions.keys())
    weights = list(transitions.values())
    return random.choices(states, weights=weights, k=1)[0]


def random_walk(start: str, steps: int) -> list[str]:
    """
    Simulate a random walk.

    Parameters
    ----------
    start : str
        Initial state ('N', 'S', 'E', or 'W').
    steps : int
        Number of transitions to perform.

    Returns
    -------
    list[str]
        Full path including the starting state (length = steps + 1).
    """
    path = [start]
    current = start
    for _ in range(steps):
        current = next_state(current)
        path.append(current)
    return path


def print_transition_matrix() -> None:
    """Pretty-print the transition matrix."""
    print("Transition Matrix (rows = from, cols = to):")
    header = "       " + "   ".join(f"{s:>4}" for s in STATES)
    print(header)
    print("      " + "-" * (len(header) - 6))
    for from_state in STATES:
        row = f"  {from_state}  |"
        for to_state in STATES:
            prob = TRANSITION_MATRIX[from_state][to_state]
            row += f"  {prob:.2f}"
        print(row)
    print()


def print_walk_statistics(path: list[str]) -> None:
    """Print visit frequency statistics for a completed walk."""
    counts = Counter(path)
    total = len(path)
    print("Visit Statistics:")
    print(f"  {'State':<8} {'Count':>6}  {'Freq':>8}  {'Expected':>10}")
    print("  " + "-" * 38)
    for state in STATES:
        count = counts.get(state, 0)
        freq = count / total
        print(f"  {state:<8} {count:>6}  {freq:>8.4f}  {'0.2500':>10}")
    print()


def print_transition_counts(path: list[str]) -> None:
    """Print observed transition counts from the walk."""
    transition_counts: dict[str, Counter] = {s: Counter() for s in STATES}
    for i in range(len(path) - 1):
        transition_counts[path[i]][path[i + 1]] += 1

    print("Observed Transition Counts (rows = from, cols = to):")
    header = "       " + "   ".join(f"{s:>5}" for s in STATES)
    print(header)
    print("      " + "-" * (len(header) - 6))
    for from_state in STATES:
        row = f"  {from_state}  |"
        for to_state in STATES:
            row += f"  {transition_counts[from_state][to_state]:>4}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    steps = 10_000
    start = random.choice(STATES)

    print("=" * 50)
    print("       Markov Chain — 2D Random Walk")
    print("=" * 50)
    print(f"  States      : {STATES}")
    print(f"  Start state : {start}")
    print(f"  Steps       : {steps:,}")
    print()

    print_transition_matrix()

    path = random_walk(start, steps)

    # Show the first 20 steps
    preview = " → ".join(path[:21]) + (" → ..." if len(path) > 21 else "")
    print(f"First 20 steps:\n  {preview}\n")

    print_walk_statistics(path)
    print_transition_counts(path)

    print("Done. The empirical frequencies should each be close to 0.25,")
    print("confirming the uniform stationary distribution of this chain.")


if __name__ == "__main__":
    main()