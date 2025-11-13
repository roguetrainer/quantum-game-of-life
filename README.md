# Quantum Game of Life Hamiltonian in PennyLane

This repository contains a complete implementation of the Quantum Game of Life (QGoL) Hamiltonian using PennyLane, based on the paper by David Faux (2019) "The semi-quantum game of life" ([arXiv:1902.07835](https://arxiv.org/abs/1902.07835)).

Another repo, [/roguetrainer/quantum-game-of-life-3x](https://github.com/roguetrainer/quantum-game-of-life-3x) contains a less sophisticated extension of the GoL, implemented in three different languages in a horse-race to compare their strenghts: Python, F# & Q#.


![QGL](./img/Q-CGL.png)

## Overview

The Quantum Game of Life extends Conway's classic cellular automaton to the quantum realm, where cells exist in superposition states between alive (|1⟩) and dead (|0⟩). The system evolves under a Hamiltonian:

```
H = Σᵢ (bᵢ + bᵢ†) · (Nᵢ⁽²⁾ + Nᵢ⁽³⁾)
```

Where:
- **(bᵢ + bᵢ†)** is the state flip operator, mapped to Pauli X
- **Nᵢ⁽ᵏ⁾** are neighbor counting projectors that equal 1 when site i has exactly k alive neighbors

See also: more detail on [the Hamiltonian](./docs/QGL-HAMILTONIAN.md), which is a [spin-chain](https://en.wikipedia.org/wiki/Spin_chain) Hamiltonian that is engineered to simulate a reversible cellular automaton rule, which mimics the complexity of Conway's Game of Life (GoL) over a continuous quantum evolution.

See [here](./docs/QGL-HAMILTONIAN-1D.md) for a conceptual Python implementation of the 1D case, using PennyLane. That simplified example uses a 1D chain with nearest-neighbour interaction. 

## Key Operator Mappings

### 1. State Flip Operator
```
bᵢ + bᵢ† ∝ σₓⁱ = qml.PauliX(i)
```

### 2. Number Operator
```
nⱼ = b†ⱼbⱼ = ½(I - Zⱼ) = ½(qml.Identity(j) - qml.PauliZ(j))
```
Projects onto the |1⟩ (alive) state.

### 3. Empty Operator
```
1 - nⱼ = ½(I + Zⱼ) = ½(qml.Identity(j) + qml.PauliZ(j))
```
Projects onto the |0⟩ (dead) state.

## Files Included

### 1. `quantum_game_of_life.py`
Complete implementation of the QGoL Hamiltonian with:
- **QuantumGameOfLife class**: Full implementation for arbitrary grid sizes
- **Neighbor counting projectors**: Build Nᵢ⁽ᵏ⁾ operators from Pauli terms
- **Hamiltonian construction**: Assemble the complete H
- **Time evolution**: Both Trotter and exact evolution methods
- **Example usage**: Demonstrates a 2×2 grid simulation

### 2. `simple_qgol_tutorial.py`
Step-by-step tutorial demonstrating:
- How to map QGoL operators to Pauli operators
- Building a simple 2-qubit Hamiltonian
- Simulating evolution with `TrotterProduct`
- Simulating evolution with exact methods
- Understanding neighbor counting projectors

## Installation

```bash
pip install pennylane numpy
```

## Quick Start

### Simple Tutorial
```bash
python simple_qgol_tutorial.py
```

This will walk you through the basics of constructing and simulating a QGoL Hamiltonian.

### Full Implementation
```python
from quantum_game_of_life import QuantumGameOfLife
import pennylane as qml
from pennylane import numpy as np

# Create a 2×2 grid
qgol = QuantumGameOfLife(grid_size=(2, 2), periodic=True)

# Build the Hamiltonian
H = qgol.build_hamiltonian()

# Prepare initial state
initial_state = np.zeros(2**qgol.n_qubits)
initial_state[0] = 1.0  # All cells dead
initial_state = initial_state / np.linalg.norm(initial_state)

# Evolve the system
time = 1.0
n_steps = 10
probs = qgol.evolve_trotter(H, time, n_steps, initial_state)
```

## How It Works

### Step 1: Constructing Neighbor Projectors

For a site with neighbors {1, 2, ..., n}, the projector Nᵢ⁽ᵏ⁾ projects onto states where exactly k neighbors are alive:

```
Nᵢ⁽ᵏ⁾ = Σ (over k-subsets S) Π_{j∈S} nⱼ · Π_{j∉S} (1-nⱼ)
```

When expanded using nⱼ = ½(I - Z), this becomes a sum of 2^n Pauli strings for each k-subset.

### Step 2: Building the Full Hamiltonian

For each site i:
1. Build Nᵢ⁽²⁾ (2 alive neighbors projector)
2. Build Nᵢ⁽³⁾ (3 alive neighbors projector)
3. Multiply by Xᵢ (state flip operator)
4. Add to the Hamiltonian

The result is a massive sum of Pauli terms that implements the QGoL rules.

### Step 3: Time Evolution

**Option A: Trotterization** (for larger systems)
```python
qml.TrotterProduct(H, t, n=n_steps)
```
Approximates exp(-iHt) using the Trotter-Suzuki decomposition.

**Option B: Exact Evolution** (for small systems)
```python
qml.ApproxTimeEvolution(H, t, n=1)
```
On simulators, computes the matrix exponential directly (more accurate but computationally expensive).

## Mathematical Details

### Hamiltonian Structure

The QGoL Hamiltonian encodes the Game of Life rules:
- **Birth**: Dead cell (|0⟩) with 3 neighbors → alive (|1⟩)
- **Survival**: Alive cell (|1⟩) with 2-3 neighbors → stays alive
- **Death**: Otherwise → dead

In quantum form:
```
H = Σᵢ Xᵢ ⊗ (Nᵢ⁽²⁾ + Nᵢ⁽³⁾)
```

When acting on state |ψ⟩:
- If site i has 2 or 3 alive neighbors, Xᵢ flips the state
- This implements state transitions under unitary evolution

### Number of Terms

For a grid with N qubits where each has m neighbors:
- Each projector Nᵢ⁽ᵏ⁾ has C(m,k) · 2^m terms
- Total terms: N · [C(m,2)·2^m + C(m,3)·2^m]

For a 2×2 grid with periodic boundaries (m=3):
- 4 sites × [3·8 + 1·8] = 128 terms

## Example Output

```
======================================================================
Quantum Game of Life Hamiltonian in PennyLane
======================================================================

1. Creating 2x2 QGoL grid with periodic boundaries...
   Number of qubits: 4

2. Building QGoL Hamiltonian H = Σᵢ Xᵢ · (Nᵢ⁽²⁾ + Nᵢ⁽³⁾)...
   Hamiltonian constructed with 128 terms

3. Preparing initial state...

4. Evolving with Trotterization...
   Evolution time: 1.0
   Trotter steps: 10
   Final state probabilities (top 5):
     |0000⟩: 0.9089
     |0101⟩: 0.0910
     ...
```

## Key Concepts

### Pauli Decomposition
Every quantum operator can be written as a linear combination of Pauli operators. The QGoL Hamiltonian is constructed entirely from:
- **I** (Identity)
- **X** (Pauli X)
- **Z** (Pauli Z)

### Tensor Products
Multi-qubit operators are built using tensor products (⊗ or `@` in PennyLane):
```python
qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliZ(2)
```
This applies X to qubit 0 and Z to qubits 1 and 2.

### Time Evolution
Unitary evolution under Hamiltonian H:
```
U(t) = exp(-iHt)
```

Trotterization approximates this as:
```
U(t) ≈ [exp(-iH₁Δt)·exp(-iH₂Δt)·...]^n
```
where H = H₁ + H₂ + ... and Δt = t/n.

## Limitations

- **Exponential scaling**: The state vector has 2^N components
- **Term explosion**: The Hamiltonian has O(N·2^m) terms
- **Practical limit**: ~10-15 qubits for simulation

For larger systems, consider:
- Tensor network methods
- Quantum hardware (if available)
- Approximate methods

## References

1. Faux, D. (2019). "The semi-quantum game of life." *arXiv preprint* arXiv:1902.07835. [Link](https://arxiv.org/abs/1902.07835)

2. PennyLane Documentation: [Time Evolution](https://docs.pennylane.ai/en/stable/code/qml_templates.html#time-evolution)

3. Conway, J. (1970). "The Game of Life." *Scientific American*

## Citation

If you use this code, please cite:

**The original paper:**
```bibtex
@misc{faux2019semiquantum,
      title={The semi-quantum game of life}, 
      author={David Faux},
      year={2019},
      eprint={1902.07835},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/1902.07835}
}
```

**This implementation:**
```bibtex
@software{qgol_pennylane,
      title={Quantum Game of Life Hamiltonian in PennyLane},
      author={Your Name},
      year={2025},
      note={Implementation based on Faux (2019)}
}
```

## License

This code is provided for educational and research purposes. The implementation follows the mathematical framework described in Faux (2019).

## Contributing

Feel free to extend this implementation:
- Add visualization tools
- Implement different lattice geometries
- Optimize the Hamiltonian construction
- Add support for modified QGoL rules

## Contact

For questions about the implementation, please open an issue or contact the author.

---

**Happy quantum computing! 🚀🎮**
