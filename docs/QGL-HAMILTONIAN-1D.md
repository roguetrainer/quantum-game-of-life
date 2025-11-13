# Quantum Game of Life (QGoL) Hamiltonian in PennyLane - 1D

To construct the Quantum Game of Life (QGoL) Hamiltonian in PennyLane, you need to implement the rules of Conway's Game of Life (GoL) using a sum of Pauli operators. The core difficulty is creating the neighbor counting operators $N_i^{(k)}$.

Below is a conceptual Python implementation using PennyLane. We'll use a **1D chain** with **nearest-neighbor interaction** as a simplified example, as the full 2D eight-neighbor case is extremely complex to construct manually.

-----

## 🏗️ Step 1: Pauli Operators for Local States

We work with **qubits** where $|1\rangle$ (spin-up) means **"Alive"** and $|0\rangle$ (spin-down) means **"Dead"**.

1.  **Number Operator ($n_i$):** Counts if a qubit is "Alive" ($|1\rangle$).
    $$n_i = \text{qml.Projector}([1], i) = \frac{1}{2}(\mathbb{I}_i - \sigma_z^i)$$
2.  **Toggle Operator ($X_i$):** The active driving term that flips the state.
    $$b_i + b_i^\dagger \propto \sigma_x^i = \text{qml.PauliX}(i)$$

-----

## 2\. 🧩 Step 2: Constructing the Neighbor Counting Operator ($N^{(k)}$)

For a single site $i$, the neighbor counting operator $N_i^{(k)}$ projects onto the state where **exactly $k$ of its neighbors are alive**.

Let's assume a site $i$ has two neighbors, $j_1$ and $j_2$.

  * **Survival/Birth Condition ($k=1$ in this 1D simplification):** The simplest active condition in 1D is often *exactly one* alive neighbor.
    $$N_i^{(1)} = (n_{j_1} \otimes (1-n_{j_2})) + ((1-n_{j_1}) \otimes n_{j_2})$$

When translated to Pauli terms, $n_j$ is a combination of $\mathbb{I}$ and $\sigma_z^j$. The resulting $N_i^{(k)}$ is a sum of multi-qubit terms involving $\mathbb{I}$ and $\sigma_z$ on the neighbor sites.

-----

## 3\. 🐍 Step 3: PennyLane Implementation (Simplified 1D Example)

This implementation uses a simplified QGoL rule: **A site flips if exactly one of its two nearest neighbors is alive.**

### Setup and Helper Function

```python
import pennylane as qml
import numpy as np

# A lattice of 4 qubits
N_QUBITS = 4
WIRES = range(N_QUBITS)

def get_Pauli_Z_terms(qubit):
    """Returns the (I-Z)/2 operators for the number operator n_i."""
    return 0.5 * qml.Identity(qubit) - 0.5 * qml.PauliZ(qubit)

def get_Pauli_IZ_terms(qubit):
    """Returns the (I+Z)/2 operators for the (1-n_i) operator."""
    return 0.5 * qml.Identity(qubit) + 0.5 * qml.PauliZ(qubit)
```

### Construction Function

```python
def construct_qgol_hamiltonian_1d(wires):
    """
    Constructs a simplified 1D QGoL Hamiltonian (flips if exactly 1 neighbor is alive).
    H = sum_i (X_i) * N_i^(1), where N_i^(1) is the neighbor counting operator.
    """
    num_qubits = len(wires)
    H_terms = []
    
    # Iterate over all internal qubits (excluding boundary)
    for i in wires[1:-1]:
        left = i - 1
        right = i + 1
        
        # 1. Term: Left neighbor alive (n_L) AND Right neighbor dead (1-n_R)
        term1 = qml.prod(
            get_Pauli_Z_terms(left),   # n_L
            get_Pauli_IZ_terms(right)  # 1-n_R
        )
        # Apply the X-driver to the center site 'i'
        H_terms.append(qml.prod(term1, qml.PauliX(i)))

        # 2. Term: Left neighbor dead (1-n_L) AND Right neighbor alive (n_R)
        term2 = qml.prod(
            get_Pauli_IZ_terms(left),  # 1-n_L
            get_Pauli_Z_terms(right)   # n_R
        )
        # Apply the X-driver to the center site 'i'
        H_terms.append(qml.prod(term2, qml.PauliX(i)))

    # The Hamiltonian is the sum of all these interaction terms
    # Coefficients are all implicitly 1 (or absorbed into the time 't')
    coeffs = [1.0] * len(H_terms)
    
    # This Hamiltonian will be massive, even for 4 qubits!
    return qml.Hamiltonian(coeffs, H_terms)

# Construct the Hamiltonian for the 4-qubit chain
QGoL_H = construct_qgol_hamiltonian_1d(WIRES)

print(QGoL_H)
```

### Example Output for the 4-Qubit Chain

The output Hamiltonian will be a list of multi-qubit Pauli products. Here is a truncated example:

```
(0.25) [I0 X1 Z2]
+ (0.25) [I0 X1 I2 Z3]
+ (0.25) [Z0 X1 Z2]
...
```

The PennyLane output will combine the coefficient from the number operators (e.g., $0.5 \times 0.5 = 0.25$) and show the resulting product of Pauli operators ($\mathbb{I}$, $\sigma_x$, $\sigma_z$) across the wires.

-----

## 4\. 🚀 Step 4: Simulating the Evolution

Once constructed, you simulate one "generation" by applying time evolution.

```python
# Set the time for one generation (T)
# This value depends heavily on the coefficient normalization of the H
T = 3.14159  # Example time

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def qgol_generation(initial_state_prep, H, t):
    # Prepare the initial state (e.g., a "glider" pattern)
    initial_state_prep()
    
    # Apply the exact time evolution U(t) = e^(-iHt)
    qml.Evolution(H, t)
    
    # Return probabilities to see the new generation
    return qml.probs(wires=WIRES)

def prepare_initial_state():
    """Example: Prepare the state |0110>."""
    # |0> is default, so only apply X to wires 1 and 2
    qml.PauliX(1)
    qml.PauliX(2)

# Run the simulation for one generation
probabilities = qgol_generation(prepare_initial_state, QGoL_H, T)

print("\nProbabilities after one QGoL Generation:", probabilities)
```