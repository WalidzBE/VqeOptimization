"""
TFIM 6 qubit — confronto energia esatta vs window ansatz VQE

Requisiti:
pip install qiskit qiskit-aer scipy numpy
"""

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit_aer.primitives import Estimator


# ============================================================
# TFIM HAMILTONIAN
# H = -J Σ Z_i Z_{i+1} - h Σ X_i
# ============================================================

def tfim_operator(n, J=1.0, h=0.7):
    terms = []

    # ZZ interaction
    for i in range(n - 1):
        s = ["I"] * n
        s[n - 1 - i] = "Z"
        s[n - 1 - (i + 1)] = "Z"
        terms.append(("".join(s), -J))

    # transverse field X
    for i in range(n):
        s = ["I"] * n
        s[n - 1 - i] = "X"
        terms.append(("".join(s), -h))

    return SparsePauliOp.from_list(terms)


# ============================================================
# EXACT GROUND ENERGY
# ============================================================

def exact_ground_energy(H):
    mat = Operator(H).data
    eigvals = np.linalg.eigvalsh(mat)
    return np.min(eigvals)


# ============================================================
# WINDOW ANSATZ (LAGRANGE FRIENDLY)
# ============================================================

def tfim_window_ansatz(n_qubits, window=5, layers=2):
    """
    Sliding window TFIM-inspired ansatz.
    Favorisce compressibilità hardware.
    """

    if window > n_qubits:
        window = n_qubits

    qc = QuantumCircuit(n_qubits)

    gamma = ParameterVector("γ", layers)
    beta = ParameterVector("β", layers)

    starts = range(0, n_qubits - window + 1)

    for l in range(layers):
        for s in starts:
            qs = list(range(s, s + window))

            # ZZ chain nella finestra
            for i in range(len(qs) - 1):
                a, b = qs[i], qs[i + 1]
                qc.cx(a, b)
                qc.rz(2 * gamma[l], b)
                qc.cx(a, b)

            # X field
            for q in qs:
                qc.rx(2 * beta[l], q)

    return qc


# ============================================================
# VQE ROUTINE
# ============================================================

def vqe_energy(ansatz, H, maxiter=500):
    estimator = Estimator()

    def objective(x):
        job = estimator.run(ansatz, H, parameter_values=x)
        return job.result().values[0]

    # inizializzazione casuale
    x0 = np.random.uniform(0, 2 * np.pi, len(ansatz.parameters))

    res = minimize(
        objective,
        x0,
        method="COBYLA",
        options={"maxiter": maxiter}
    )

    return res.fun


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    n = 6
    J = 1.0
    h = 0.7

    print("\n==============================")
    print("TFIM WINDOW ANSATZ TEST (6 QUBIT)")
    print("==============================")

    # --- Hamiltonian ---
    print("\nBuilding TFIM Hamiltonian...")
    H = tfim_operator(n, J, h)

    # --- Exact energy ---
    print("\nComputing exact ground energy...")
    E_exact = exact_ground_energy(H)
    print("Exact ground energy:", E_exact)

    # --- Window ansatz ---
    print("\nBuilding window ansatz...")
    ansatz = tfim_window_ansatz(n, window=5, layers=1)

    print("Number of parameters:", len(ansatz.parameters))
    print("Circuit depth:", ansatz.depth())

    # --- VQE ---
    print("\nRunning VQE optimization (COBYLA)...")
    E_vqe = vqe_energy(ansatz, H)

    # --- Results ---
    print("\n==============================")
    print("RESULTS")
    print("==============================")

    print("Exact energy:", E_exact)
    print("VQE energy:", E_vqe)
    print("Absolute error:", abs(E_vqe - E_exact))
    print("Relative error:", abs(E_vqe - E_exact) / abs(E_exact))
