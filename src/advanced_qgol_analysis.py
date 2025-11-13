"""
Advanced QGoL Example: Visualization and Analysis
==================================================

This script demonstrates:
1. Simulating QGoL evolution over multiple time steps
2. Analyzing the "quantum cloud" emergence
3. Visualizing the probability distribution over time
4. Comparing classical vs quantum evolution

Run this after understanding the basics from simple_qgol_tutorial.py
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from quantum_game_of_life import QuantumGameOfLife


def analyze_liveness_distribution(probs, n_qubits):
    """
    Calculate the liveness distribution from probability vector.
    
    In the QGoL paper, the "quantum cloud" is characterized by its
    liveness distribution with mean ≈ 0.348.
    
    Args:
        probs: Probability vector for all 2^n basis states
        n_qubits: Number of qubits
        
    Returns:
        mean_liveness: Average number of alive cells
        std_liveness: Standard deviation
    """
    liveness_values = []
    
    for state_idx, prob in enumerate(probs):
        if prob > 1e-10:  # Only consider non-zero probabilities
            # Count number of 1s in binary representation (alive cells)
            n_alive = bin(state_idx).count('1')
            liveness_values.extend([n_alive] * int(prob * 10000))  # Weight by probability
    
    mean_liveness = np.mean(liveness_values) if liveness_values else 0
    std_liveness = np.std(liveness_values) if liveness_values else 0
    
    return mean_liveness, std_liveness


def visualize_grid_probabilities(probs, grid_size, title="QGoL State"):
    """
    Visualize the marginal probability of each cell being alive.
    
    Args:
        probs: Full probability vector
        grid_size: Tuple (rows, cols)
        title: Plot title
    """
    rows, cols = grid_size
    n_qubits = rows * cols
    
    # Calculate marginal probability for each qubit
    marginal_probs = np.zeros(n_qubits)
    
    for state_idx, prob in enumerate(probs):
        # Get binary representation
        binary = format(state_idx, f'0{n_qubits}b')
        for qubit in range(n_qubits):
            if binary[qubit] == '1':
                marginal_probs[qubit] += prob
    
    # Reshape to grid
    grid_probs = marginal_probs.reshape(rows, cols)
    
    # Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(grid_probs, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(label='P(alive)')
    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Add probability values
    for i in range(rows):
        for j in range(cols):
            plt.text(j, i, f'{grid_probs[i, j]:.2f}', 
                    ha='center', va='center', color='black', fontsize=12)
    
    plt.tight_layout()
    return grid_probs


def simulate_qgol_trajectory(qgol, H, initial_state, time_points):
    """
    Simulate QGoL evolution at multiple time points.
    
    Args:
        qgol: QuantumGameOfLife instance
        H: Hamiltonian
        initial_state: Initial quantum state
        time_points: List of times to evaluate
        
    Returns:
        trajectory: List of probability vectors at each time point
    """
    trajectory = []
    
    for t in time_points:
        probs = qgol.evolve_trotter(H, t, n_steps=20, initial_state=initial_state)
        trajectory.append(probs)
    
    return trajectory


def plot_liveness_evolution(trajectory, time_points, n_qubits):
    """
    Plot the evolution of mean liveness over time.
    
    This shows how the system evolves toward the "quantum cloud"
    with characteristic mean liveness ≈ 0.348 (from the paper).
    """
    mean_liveness_values = []
    std_liveness_values = []
    
    for probs in trajectory:
        mean, std = analyze_liveness_distribution(probs, n_qubits)
        mean_liveness_values.append(mean)
        std_liveness_values.append(std)
    
    plt.figure(figsize=(10, 4))
    
    # Plot mean liveness
    plt.subplot(1, 2, 1)
    plt.plot(time_points, mean_liveness_values, 'b-o', linewidth=2)
    plt.axhline(y=0.348, color='r', linestyle='--', label='Paper value (≈0.348)')
    plt.xlabel('Time')
    plt.ylabel('Mean Liveness')
    plt.title('Evolution of Mean Liveness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot std liveness
    plt.subplot(1, 2, 2)
    plt.plot(time_points, std_liveness_values, 'g-o', linewidth=2)
    plt.axhline(y=0.0071, color='r', linestyle='--', label='Paper value (≈0.0071)')
    plt.xlabel('Time')
    plt.ylabel('Std Liveness')
    plt.title('Evolution of Liveness Std Dev')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()


def main():
    """Main analysis and visualization."""
    
    print("=" * 70)
    print("Advanced QGoL Analysis and Visualization")
    print("=" * 70)
    
    # Setup
    grid_size = (2, 2)
    qgol = QuantumGameOfLife(grid_size=grid_size, periodic=True)
    H = qgol.build_hamiltonian()
    
    print(f"\nGrid size: {grid_size[0]}×{grid_size[1]} = {qgol.n_qubits} qubits")
    print(f"Hamiltonian terms: {len(H.ops)}")
    
    # Prepare initial state: superposition of multiple basis states
    print("\n1. Preparing initial superposition state...")
    initial_state = np.zeros(2**qgol.n_qubits)
    # Create a non-trivial initial state with multiple components
    initial_state[0] = 0.5   # |0000⟩
    initial_state[5] = 0.5   # |0101⟩
    initial_state[10] = 0.5  # |1010⟩
    initial_state[15] = 0.5  # |1111⟩
    initial_state = initial_state / np.linalg.norm(initial_state)
    
    print(f"Initial state prepared with {np.count_nonzero(initial_state)} non-zero amplitudes")
    
    # Simulate trajectory
    print("\n2. Simulating time evolution...")
    time_points = np.linspace(0, 5, 11)
    trajectory = simulate_qgol_trajectory(qgol, H, initial_state, time_points)
    
    print(f"Computed {len(time_points)} time steps")
    
    # Analyze liveness evolution
    print("\n3. Analyzing liveness distribution...")
    print(f"\n{'Time':<10} {'Mean Liveness':<15} {'Std Liveness':<15}")
    print("-" * 40)
    for t, probs in zip(time_points, trajectory):
        mean, std = analyze_liveness_distribution(probs, qgol.n_qubits)
        print(f"{t:<10.2f} {mean:<15.4f} {std:<15.4f}")
    
    # Compare with paper values
    final_mean, final_std = analyze_liveness_distribution(trajectory[-1], qgol.n_qubits)
    paper_mean = 0.348
    paper_std = 0.0071
    
    print(f"\n📊 Comparison with paper (Faux 2019):")
    print(f"   Paper mean liveness: {paper_mean}")
    print(f"   Our final mean: {final_mean:.4f}")
    print(f"   Paper std liveness: {paper_std}")
    print(f"   Our final std: {final_std:.4f}")
    print(f"\nNote: Exact values depend on grid size, boundary conditions,")
    print(f"      initial state, and evolution time. The paper used larger")
    print(f"      grids and longer evolution times to reach the quantum cloud.")
    
    # Visualizations
    print("\n4. Generating visualizations...")
    
    # Plot initial state
    visualize_grid_probabilities(initial_state**2, grid_size, 
                                 title="Initial State Probabilities")
    plt.savefig('/mnt/user-data/outputs/qgol_initial_state.png', dpi=150)
    print("   ✓ Saved: qgol_initial_state.png")
    
    # Plot final state
    visualize_grid_probabilities(trajectory[-1], grid_size,
                                 title=f"Final State (t={time_points[-1]:.1f})")
    plt.savefig('/mnt/user-data/outputs/qgol_final_state.png', dpi=150)
    print("   ✓ Saved: qgol_final_state.png")
    
    # Plot liveness evolution
    plot_liveness_evolution(trajectory, time_points, qgol.n_qubits)
    plt.savefig('/mnt/user-data/outputs/qgol_liveness_evolution.png', dpi=150)
    print("   ✓ Saved: qgol_liveness_evolution.png")
    
    print("\n" + "=" * 70)
    print("Analysis complete! Check the output directory for visualizations.")
    print("=" * 70)


if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        main()
    except ImportError:
        print("Error: matplotlib is required for visualization.")
        print("Install with: pip install matplotlib")
        
        # Run without visualization
        print("\nRunning analysis without visualization...")
        grid_size = (2, 2)
        qgol = QuantumGameOfLife(grid_size=grid_size, periodic=True)
        H = qgol.build_hamiltonian()
        
        initial_state = np.zeros(2**qgol.n_qubits)
        initial_state[0] = 0.5
        initial_state[5] = 0.5
        initial_state[10] = 0.5
        initial_state[15] = 0.5
        initial_state = initial_state / np.linalg.norm(initial_state)
        
        time_points = [0, 1, 2, 3, 4, 5]
        print(f"\n{'Time':<10} {'Mean Liveness':<15}")
        print("-" * 25)
        
        for t in time_points:
            probs = qgol.evolve_trotter(H, t, n_steps=20, initial_state=initial_state)
            mean, _ = analyze_liveness_distribution(probs, qgol.n_qubits)
            print(f"{t:<10.2f} {mean:<15.4f}")
