🎮⚛️ Quantum Game of Life: Conway meets Hamiltonian Dynamics
Just implemented a quantum cellular automaton version of Conway's Game of Life using PennyLane! Instead of discrete time-step rules, cells evolve under continuous Hamiltonian dynamics.
Key approach: Map GoL rules to a quantum Hamiltonian:

Cell flip → Pauli X
Neighbor counting → Projectors from ½(I - Z)
Result: H = Σᵢ Xᵢ ⊗ (Nᵢ⁽²⁾ + Nᵢ⁽³⁾)

Quantum superposition and entanglement create behavior impossible in classical GoL - cells exist in superposed states and quantum interference drives complex dynamics.
Built complete tutorials + Jupyter notebooks for exploring quantum cellular automata with PennyLane.
#QuantumComputing #PennyLane #CellularAutomata #QuantumAlgorithms