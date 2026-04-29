import numpy as np

transition_matrix = np.array([
    [0.5, 0.2, 0.3],
    [0.1, 0.8, 0.1],
    [0.2, 0.4, 0.4]
])

def get_stationary_distribution(transition_matrix):
    # Get the number of states
    num_states = transition_matrix.shape[0]

    # Create a matrix A and a vector b for the linear system A * x = b
    A = np.transpose(transition_matrix) - np.eye(num_states)
    b = np.zeros(num_states)

    # Add the constraint that the sum of the probabilities must be 1
    A = np.vstack([A, np.ones(num_states)])
    b = np.append(b, 1)

    # Solve the linear system to find the stationary distribution
    stationary_distribution = np.linalg.lstsq(A, b, rcond=None)[0]

    return stationary_distribution


def iterate_until_stationary(transition_matrix, initial_distribution=None, tolerance=1e-8, max_iterations=100):
    num_states = transition_matrix.shape[0]

    # If no initial distribution is given, start evenly
    if initial_distribution is None:
        current_distribution = np.ones(num_states) / num_states
    else:
        current_distribution = np.array(initial_distribution, dtype=float)

    stationary_distribution = get_stationary_distribution(transition_matrix)

    print("Stationary distribution:", stationary_distribution)
    print()

    print(f"Iteration 0: {current_distribution}")

    for i in range(1, max_iterations + 1):
        current_distribution = current_distribution @ transition_matrix
        print(f"Iteration {i}: {current_distribution}")

        # Stop when close enough to the stationary distribution
        if np.allclose(current_distribution, stationary_distribution, atol=tolerance):
            print("\nConverged to the stationary distribution.")
            break
    else:
        print("\nDid not fully converge within the max number of iterations.")

    return current_distribution


# Example usage
iterate_until_stationary(
    transition_matrix,
    initial_distribution=[1, 0, 0],   # start completely in state 1
    tolerance=1e-8,
    max_iterations=100
)