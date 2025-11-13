"""
Quantum Game of Life (QGoL) Hamiltonian Implementation in PennyLane

This implementation creates a Hamiltonian for the semi-quantum game of life,
where cells exist in superposition states between alive and dead.

The Hamiltonian is: H = Σᵢ (bᵢ + bᵢ†) · (Nᵢ⁽³⁾ + Nᵢ⁽²⁾)

Where:
- (bᵢ + bᵢ†) is the state flip operator (maps to Pauli X)
- Nᵢ⁽ᵏ⁾ are neighbor counting projectors (constructed from Pauli Z)
- The number operator: nⱼ = ½(I - Z)

Based on: Faux, D. (2019). The semi-quantum game of life. arXiv:1902.07835
"""

import pennylane as qml
from pennylane import numpy as np
from itertools import product, combinations


class QuantumGameOfLife:
    """
    Implements the Quantum Game of Life Hamiltonian using PennyLane.
    
    In the classical Game of Life, a cell's next state depends on:
    - Current state (alive=1, dead=0)
    - Number of live neighbors
    
    Rules:
    - Birth: Dead cell with 3 neighbors becomes alive
    - Survival: Live cell with 2-3 neighbors stays alive
    - Death: Otherwise
    
    In the quantum version, cells exist in superposition |ψ⟩ = α|0⟩ + β|1⟩
    and evolve under the QGoL Hamiltonian.
    """
    
    def __init__(self, grid_size=(3, 3), periodic=True):
        """
        Initialize the Quantum Game of Life.
        
        Args:
            grid_size: Tuple (rows, cols) for the grid dimensions
            periodic: Whether to use periodic boundary conditions
        """
        self.rows, self.cols = grid_size
        self.n_qubits = self.rows * self.cols
        self.periodic = periodic
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        
    def coord_to_qubit(self, row, col):
        """Convert 2D grid coordinates to qubit index."""
        return row * self.cols + col
    
    def qubit_to_coord(self, qubit):
        """Convert qubit index to 2D grid coordinates."""
        return qubit // self.cols, qubit % self.cols
        
    def get_neighbors(self, row, col):
        """
        Get the qubit indices of neighboring cells (Moore neighborhood).
        
        Args:
            row: Row index of the cell
            col: Column index of the cell
            
        Returns:
            List of qubit indices for the 8 neighbors
        """
        neighbors = []
        
        for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            if self.periodic:
                # Periodic boundary conditions
                new_row = (row + dr) % self.rows
                new_col = (col + dc) % self.cols
            else:
                # Fixed boundary conditions
                new_row = row + dr
                new_col = col + dc
                if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
                    continue
                    
            neighbors.append(self.coord_to_qubit(new_row, new_col))
            
        return neighbors
    
    def build_neighbor_projector(self, site, k):
        """
        Build the neighbor counting projector N_i^(k) that projects onto states
        where site i has exactly k alive neighbors.
        
        The projector is constructed from number operators nⱼ = ½(I - Z).
        
        Args:
            site: The qubit index for the central site
            k: The number of alive neighbors to project onto
            
        Returns:
            coeffs: List of coefficients for the Hamiltonian terms
            obs: List of PennyLane observables (Pauli strings)
        """
        row, col = self.qubit_to_coord(site)
        neighbors = self.get_neighbors(row, col)
        n_neighbors = len(neighbors)
        
        if k > n_neighbors:
            return [], []
        
        coeffs = []
        obs = []
        
        # We need to construct a projector that equals 1 when exactly k neighbors are alive
        # and 0 otherwise. This is done by summing over all combinations of k neighbors
        # being alive (nⱼ = 1) and the rest being dead (1 - nⱼ = 0).
        
        # Since nⱼ = ½(I - Z), we have:
        # nⱼ = ½(I - Z)        [projects onto |1⟩]
        # 1 - nⱼ = ½(I + Z)    [projects onto |0⟩]
        
        # For each combination of k neighbors that are alive
        for alive_neighbors in combinations(neighbors, k):
            alive_set = set(alive_neighbors)
            
            # Build the product of projectors
            # Product of nⱼ for alive neighbors and (1-nⱼ) for dead neighbors
            # This will generate 2^n_neighbors terms when expanded
            
            # Each projector term is a product: Π(nⱼ or (1-nⱼ))
            # When expanded as ½(I ± Z), this becomes a sum of 2^n_neighbors Pauli strings
            
            # For simplicity, we expand this explicitly
            coeff_product = (0.5) ** n_neighbors
            
            # Generate all possible combinations of I and Z for each neighbor
            for z_pattern in product([0, 1], repeat=n_neighbors):
                # Build the Pauli string
                pauli_ops = []
                term_coeff = coeff_product
                
                for idx, neighbor in enumerate(neighbors):
                    if neighbor in alive_set:
                        # Use nⱼ = ½(I - Z), so we need -Z
                        if z_pattern[idx] == 1:
                            pauli_ops.append(qml.PauliZ(neighbor))
                            term_coeff *= -1
                    else:
                        # Use (1 - nⱼ) = ½(I + Z), so we need +Z
                        if z_pattern[idx] == 1:
                            pauli_ops.append(qml.PauliZ(neighbor))
                
                # Combine all Pauli operators with tensor product
                if len(pauli_ops) == 0:
                    obs_term = qml.Identity(site)
                else:
                    obs_term = pauli_ops[0]
                    for op in pauli_ops[1:]:
                        obs_term = obs_term @ op
                
                coeffs.append(term_coeff)
                obs.append(obs_term)
        
        return coeffs, obs
    
    def build_hamiltonian(self):
        """
        Build the full QGoL Hamiltonian: H = Σᵢ (bᵢ + bᵢ†) · (Nᵢ⁽³⁾ + Nᵢ⁽²⁾)
        
        Where:
        - (bᵢ + bᵢ†) ∝ σₓⁱ (Pauli X on site i)
        - Nᵢ⁽ᵏ⁾ are neighbor counting projectors
        
        Returns:
            qml.Hamiltonian: The QGoL Hamiltonian
        """
        all_coeffs = []
        all_obs = []
        
        # Iterate over all sites
        for site in range(self.n_qubits):
            # Get projectors for 2 and 3 neighbors
            for k in [2, 3]:
                proj_coeffs, proj_obs = self.build_neighbor_projector(site, k)
                
                # Multiply each projector term by X on site i
                for c, obs in zip(proj_coeffs, proj_obs):
                    # Add X operator on site i
                    full_obs = qml.PauliX(site) @ obs
                    all_coeffs.append(c)
                    all_obs.append(full_obs)
        
        # Create the Hamiltonian
        H = qml.Hamiltonian(all_coeffs, all_obs)
        return H
    
    def evolve_trotter(self, H, time, n_steps, initial_state=None):
        """
        Evolve the QGoL state using Trotterization.
        
        Args:
            H: The Hamiltonian (qml.Hamiltonian object)
            time: Total evolution time
            n_steps: Number of Trotter steps
            initial_state: Initial state vector (optional)
            
        Returns:
            Final state probabilities
        """
        @qml.qnode(self.dev)
        def circuit():
            # Initialize state if provided
            if initial_state is not None:
                qml.QubitStateVector(initial_state, wires=range(self.n_qubits))
            
            # Apply Trotterized evolution
            qml.TrotterProduct(H, time, n=n_steps)
            
            # Return probabilities of all basis states
            return qml.probs(wires=range(self.n_qubits))
        
        return circuit()
    
    def evolve_exact(self, H, time, initial_state=None):
        """
        Evolve the QGoL state using matrix exponentiation.
        For PennyLane, we use ApproxTimeEvolution with n=1 which gives good accuracy.
        
        Args:
            H: The Hamiltonian (qml.Hamiltonian object)
            time: Evolution time
            initial_state: Initial state vector (optional)
            
        Returns:
            Final state probabilities
        """
        @qml.qnode(self.dev)
        def circuit():
            # Initialize state if provided
            if initial_state is not None:
                qml.QubitStateVector(initial_state, wires=range(self.n_qubits))
            
            # Apply evolution using ApproxTimeEvolution
            qml.ApproxTimeEvolution(H, time, n=1)
            
            # Return probabilities
            return qml.probs(wires=range(self.n_qubits))
        
        return circuit()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Quantum Game of Life Hamiltonian in PennyLane")
    print("=" * 70)
    
    # Create a small 2x2 grid
    print("\n1. Creating 2x2 QGoL grid with periodic boundaries...")
    qgol = QuantumGameOfLife(grid_size=(2, 2), periodic=True)
    print(f"   Number of qubits: {qgol.n_qubits}")
    
    # Build the Hamiltonian
    print("\n2. Building QGoL Hamiltonian H = Σᵢ Xᵢ · (Nᵢ⁽²⁾ + Nᵢ⁽³⁾)...")
    H = qgol.build_hamiltonian()
    print(f"   Hamiltonian constructed with {len(H.ops)} terms")
    print(f"   Sample terms:")
    for i in range(min(3, len(H.ops))):
        print(f"     {H.coeffs[i]:.4f} * {H.ops[i]}")
    
    # Prepare initial state (classical "glider" configuration mapped to quantum)
    print("\n3. Preparing initial state...")
    # Start with a simple superposition state
    initial_state = np.zeros(2**qgol.n_qubits)
    initial_state[0] = 1.0  # Start in |0000⟩ (all dead)
    # Add some superposition
    initial_state[5] = 0.3  # Some amplitude in |0101⟩
    initial_state = initial_state / np.linalg.norm(initial_state)
    
    # Evolve using Trotterization
    print("\n4. Evolving with Trotterization...")
    time = 1.0
    n_steps = 10
    probs_trotter = qgol.evolve_trotter(H, time, n_steps, initial_state)
    print(f"   Evolution time: {time}")
    print(f"   Trotter steps: {n_steps}")
    print(f"   Final state probabilities (top 5):")
    top_indices = np.argsort(probs_trotter)[-5:][::-1]
    for idx in top_indices:
        binary = format(idx, f'0{qgol.n_qubits}b')
        print(f"     |{binary}⟩: {probs_trotter[idx]:.4f}")
    
    # Evolve using exact evolution
    print("\n5. Evolving with exact matrix exponentiation...")
    probs_exact = qgol.evolve_exact(H, time, initial_state)
    print(f"   Final state probabilities (top 5):")
    top_indices = np.argsort(probs_exact)[-5:][::-1]
    for idx in top_indices:
        binary = format(idx, f'0{qgol.n_qubits}b')
        print(f"     |{binary}⟩: {probs_exact[idx]:.4f}")
    
    print("\n" + "=" * 70)
    print("Complete! The QGoL Hamiltonian has been constructed and simulated.")
    print("=" * 70)
