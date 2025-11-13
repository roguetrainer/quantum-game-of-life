# Quick Start Guide: QGoL Hamiltonian in PennyLane

## What You Have

Four main Python files implementing the Quantum Game of Life (QGoL) Hamiltonian:

1. **`simple_qgol_tutorial.py`** ⭐ START HERE
   - Interactive tutorial with step-by-step explanations
   - Shows operator mappings (X, Z, I)
   - Demonstrates Trotterization vs exact evolution
   - Best for learning the basics

2. **`quantum_game_of_life.py`** 🔧 MAIN IMPLEMENTATION
   - Complete `QuantumGameOfLife` class
   - Handles arbitrary grid sizes
   - Builds neighbor counting projectors
   - Two evolution methods: Trotter and exact
   - Example usage included

3. **`advanced_qgol_analysis.py`** 📊 ADVANCED FEATURES
   - Trajectory analysis over time
   - Liveness distribution calculations
   - Comparison with paper results
   - Visualization generation (requires matplotlib)

4. **`README.md`** 📖 FULL DOCUMENTATION
   - Complete mathematical background
   - Implementation details
   - Usage examples
   - References and citations

## Running the Code

### Step 1: Install Dependencies
```bash
pip install pennylane numpy
pip install matplotlib  # Optional, for visualizations
```

### Step 2: Run the Tutorial
```bash
python simple_qgol_tutorial.py
```

Expected output:
- Explanation of operator mappings
- Hamiltonian construction demo
- Time evolution comparison
- Neighbor counting example

### Step 3: Try the Full Implementation
```python
from quantum_game_of_life import QuantumGameOfLife
import numpy as np

# Create a 3×3 grid
qgol = QuantumGameOfLife(grid_size=(3, 3), periodic=True)

# Build Hamiltonian
H = qgol.build_hamiltonian()
print(f"Hamiltonian has {len(H.ops)} terms")

# Prepare initial state
initial_state = np.zeros(2**qgol.n_qubits)
initial_state[0] = 1.0  # All dead
initial_state = initial_state / np.linalg.norm(initial_state)

# Evolve
probs = qgol.evolve_trotter(H, time=1.0, n_steps=10, initial_state=initial_state)

# Analyze
for i, p in enumerate(probs[:5]):
    print(f"State {i:03b}: {p:.4f}")
```

### Step 4: Advanced Analysis
```bash
python advanced_qgol_analysis.py
```

This generates visualizations showing:
- Initial and final state probabilities
- Liveness evolution over time
- Comparison with paper results

## Key Concepts Summary

### The Hamiltonian
```
H = Σᵢ Xᵢ ⊗ (Nᵢ⁽²⁾ + Nᵢ⁽³⁾)
```

### Operator Mappings
| Quantum | Pauli | PennyLane |
|---------|-------|-----------|
| b + b† | X | `qml.PauliX(i)` |
| n = b†b | ½(I - Z) | `½(qml.Identity(i) - qml.PauliZ(i))` |
| 1 - n | ½(I + Z) | `½(qml.Identity(i) + qml.PauliZ(i))` |

### Time Evolution Methods

**Trotterization** (approximate, scalable):
```python
qml.TrotterProduct(H, time, n=n_steps)
```

**Exact** (accurate, small systems only):
```python
qml.ApproxTimeEvolution(H, time, n=1)
```

## Understanding the Code

### Grid to Qubit Mapping
A 2×2 grid maps to 4 qubits:
```
[0, 1]    →    Qubits: [0, 1, 2, 3]
[2, 3]
```

### Neighbor Relationships
With periodic boundaries, each cell has 8 neighbors (Moore neighborhood):
```
[7, 0, 1]
[6, •, 2]
[5, 4, 3]
```

### State Encoding
Binary states encode cell configurations:
- `|0000⟩` = all dead
- `|1111⟩` = all alive
- `|0101⟩` = checkerboard pattern

## Common Pitfalls

1. **Memory**: 2×2 grid = 16 states, 3×3 = 512 states, 4×4 = 65,536 states
   - Stay below ~10 qubits for practical simulation

2. **Hamiltonian Size**: Number of terms grows exponentially
   - 2×2 periodic: 128 terms
   - 3×3 periodic: 1,152 terms

3. **Evolution Time**: Choice of time parameter affects behavior
   - Start with small values (0.1 - 2.0)
   - Increase Trotter steps for better accuracy

## Next Steps

1. **Modify Grid Size**: Try different configurations
   ```python
   qgol = QuantumGameOfLife(grid_size=(3, 2), periodic=False)
   ```

2. **Custom Initial States**: Create interesting patterns
   ```python
   # Glider pattern
   initial_state[0b010001000] = 1.0  # Example
   ```

3. **Analyze Results**: Extract meaningful statistics
   ```python
   from advanced_qgol_analysis import analyze_liveness_distribution
   mean, std = analyze_liveness_distribution(probs, n_qubits)
   ```

4. **Visualize Evolution**: Track how states change over time
   ```python
   times = np.linspace(0, 5, 20)
   trajectory = [qgol.evolve_trotter(H, t, 10, initial) for t in times]
   ```

## Citation

Paper: Faux, D. (2019). "The semi-quantum game of life." arXiv:1902.07835

```bibtex
@misc{faux2019semiquantum,
      title={The semi-quantum game of life}, 
      author={David Faux},
      year={2019},
      eprint={1902.07835},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

## Troubleshooting

**Import Error**: Make sure PennyLane is installed
```bash
pip install pennylane --upgrade
```

**Out of Memory**: Reduce grid size or use fewer qubits

**Slow Execution**: 
- Reduce Trotter steps
- Use smaller grids
- Consider GPU acceleration (if available)

**JAX Warning**: Can be safely ignored for most uses
```bash
pip install jax==0.6.0 jaxlib==0.6.0  # Optional fix
```

## Support

For questions about:
- **Implementation**: See `README.md` for detailed docs
- **PennyLane**: Visit https://pennylane.ai/
- **QGoL Paper**: Read arXiv:1902.07835

---

**Ready to explore quantum cellular automata? Start with `simple_qgol_tutorial.py`!** 🚀
