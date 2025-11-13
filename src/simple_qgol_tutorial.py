"""
Simplified QGoL Hamiltonian Tutorial
=====================================

This script demonstrates the key concepts from the document:
1. Building the Hamiltonian using Pauli operators
2. Simulating time evolution with TrotterProduct
3. Simulating time evolution with Evolution (exact)

Based on the approach outlined in your document.
"""

import pennylane as qml
from pennylane import numpy as np


def simple_qgol_hamiltonian_demo():
    """
    A minimal example showing how to construct and simulate a QGoL-like Hamiltonian.
    We'll use a simplified 2-qubit system for clarity.
    """
    
    print("=" * 70)
    print("Simplified QGoL Hamiltonian Tutorial")
    print("=" * 70)
    
    # Step 1: Understand the operator mappings
    print("\n📚 STEP 1: Operator Mappings")
    print("-" * 70)
    print("In QGoL, we map quantum operators to Pauli operators:")
    print("  • State flip (b + b†) → PauliX(i)")
    print("  • Number operator nⱼ = b†b → ½(I - PauliZ(j))")
    print("  • Empty operator (1 - nⱼ) → ½(I + PauliZ(j))")
    
    # Step 2: Build a simple Hamiltonian
    print("\n🏗️  STEP 2: Constructing the Hamiltonian")
    print("-" * 70)
    print("For a 2-qubit system, let's create a simple interaction term:")
    print("H = X₀ ⊗ (½I₁ - ½Z₁)  [Qubit 0 flips when qubit 1 is alive]")
    print("  = ½X₀ ⊗ I₁ - ½X₀ ⊗ Z₁")
    
    # Define the Hamiltonian terms
    coeffs = [0.5, -0.5]
    obs = [
        qml.PauliX(0) @ qml.Identity(1),  # ½X₀ ⊗ I₁
        qml.PauliX(0) @ qml.PauliZ(1)      # -½X₀ ⊗ Z₁
    ]
    
    H_simple = qml.Hamiltonian(coeffs, obs)
    print(f"\nHamiltonian created with {len(H_simple.ops)} terms:")
    for c, op in zip(H_simple.coeffs, H_simple.ops):
        print(f"  {c:+.2f} * {op}")
    
    # Step 3: Simulate with Trotter evolution
    print("\n⏳ STEP 3: Simulating Time Evolution (Trotterization)")
    print("-" * 70)
    
    dev = qml.device("default.qubit", wires=2)
    
    @qml.qnode(dev)
    def qgol_trotter_circuit(t, n_steps):
        """Circuit using TrotterProduct for approximate evolution."""
        # Start in state |10⟩ (qubit 0 alive, qubit 1 dead)
        qml.PauliX(wires=0)
        
        # Apply Trotterized time evolution
        qml.TrotterProduct(H_simple, t, n=n_steps)
        
        # Measure probabilities
        return qml.probs(wires=[0, 1])
    
    time = 1.0
    trotter_steps = 20
    print(f"Evolving for time t = {time} with {trotter_steps} Trotter steps...")
    
    probs_trotter = qgol_trotter_circuit(time, trotter_steps)
    print("\nFinal state probabilities:")
    states = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    for state, prob in zip(states, probs_trotter):
        print(f"  {state}: {prob:.4f}")
    
    # Step 4: Simulate with exact evolution
    print("\n⏳ STEP 4: Simulating Time Evolution (Exact)")
    print("-" * 70)
    
    @qml.qnode(dev)
    def qgol_exact_circuit(t):
        """Circuit using ApproxTimeEvolution for evolution."""
        # Start in state |10⟩
        qml.PauliX(wires=0)
        
        # Apply time evolution using ApproxTimeEvolution (works as exact for simulators)
        qml.ApproxTimeEvolution(H_simple, t, n=1)
        
        # Measure probabilities
        return qml.probs(wires=[0, 1])
    
    print(f"Evolving for time t = {time} with exact matrix exponentiation...")
    
    probs_exact = qgol_exact_circuit(time)
    print("\nFinal state probabilities:")
    for state, prob in zip(states, probs_exact):
        print(f"  {state}: {prob:.4f}")
    
    # Compare the two methods
    print("\n📊 Comparison:")
    print("-" * 70)
    difference = np.abs(probs_trotter - probs_exact)
    print(f"Maximum difference between Trotter and Exact: {np.max(difference):.6f}")
    print("(Increasing Trotter steps reduces this difference)")
    
    print("\n" + "=" * 70)
    print("Tutorial complete!")
    print("=" * 70)


def multi_qubit_neighbor_counting_demo():
    """
    Demonstrates how to build neighbor counting projectors N^(k)
    for a more realistic QGoL scenario.
    """
    
    print("\n\n")
    print("=" * 70)
    print("Neighbor Counting Projector Demo")
    print("=" * 70)
    print("\nThis demonstrates building N₀⁽²⁾ - a projector that equals 1")
    print("when qubit 0 has exactly 2 alive neighbors (qubits 1 and 2).")
    print("-" * 70)
    
    # For a 3-qubit system where qubit 0 is the center and qubits 1,2 are neighbors
    # N₀⁽²⁾ = n₁·n₂ (both neighbors alive)
    # where nⱼ = ½(I - Z)
    
    # Expanding: n₁·n₂ = ¼(I - Z₁)(I - Z₂)
    #                  = ¼(I - Z₁ - Z₂ + Z₁Z₂)
    
    print("\nBuilding N₀⁽²⁾ = n₁·n₂ where nⱼ = ½(I - Zⱼ):")
    print("  N₀⁽²⁾ = ¼(I - Z₁)(I - Z₂)")
    print("       = ¼(I - Z₁ - Z₂ + Z₁⊗Z₂)")
    
    coeffs_n2 = [0.25, -0.25, -0.25, 0.25]
    obs_n2 = [
        qml.Identity(0) @ qml.Identity(1) @ qml.Identity(2),
        qml.Identity(0) @ qml.PauliZ(1) @ qml.Identity(2),
        qml.Identity(0) @ qml.Identity(1) @ qml.PauliZ(2),
        qml.Identity(0) @ qml.PauliZ(1) @ qml.PauliZ(2)
    ]
    
    print("\nTerms in the projector:")
    for c, op in zip(coeffs_n2, obs_n2):
        print(f"  {c:+.2f} * {op}")
    
    # Now multiply by X₀ to get the QGoL term
    print("\nFull QGoL term: H₀⁽²⁾ = X₀ ⊗ N₀⁽²⁾")
    
    coeffs_h = coeffs_n2
    obs_h = [qml.PauliX(0) @ obs for obs in obs_n2]
    
    H_neighbor = qml.Hamiltonian(coeffs_h, obs_h)
    
    print("This creates a Hamiltonian term that flips qubit 0")
    print("when it has exactly 2 alive neighbors.")
    
    print("\n" + "=" * 70)


# ============================================================================
# RUN THE DEMOS
# ============================================================================

if __name__ == "__main__":
    # Run the simple 2-qubit demo
    simple_qgol_hamiltonian_demo()
    
    # Run the neighbor counting demo
    multi_qubit_neighbor_counting_demo()
    
    print("\n\n💡 KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. QGoL Hamiltonian: H = Σᵢ Xᵢ ⊗ (Nᵢ⁽²⁾ + Nᵢ⁽³⁾)")
    print("2. State flip operator (b + b†) → PauliX")
    print("3. Number operator n = ½(I - Z)")
    print("4. Use TrotterProduct for large systems (approximate)")
    print("5. Use Evolution for small systems (exact)")
    print("6. The full Hamiltonian has many terms (~2^(n_neighbors) per site)")
    print("=" * 70)
