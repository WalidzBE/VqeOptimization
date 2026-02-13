"""
TFIM 6 qubit — confronto energia esatta vs window ansatz VQE (OPTIMIZED)

Migliorie:
- window ansatz layer=1 (compressible)
- initial state |+>
- Powell optimizer (molto efficace per pochi parametri)
- multi-start optimization
- inizializzazione parametri intelligente

Requisiti:
pip install qiskit qiskit-aer scipy numpy
"""

import numpy as np
from scipy.optimize import minimize

from qiskit_aer import AerSimulator
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
# WINDOW ANSATZ (LAGRANGE FRIENDLY + INITIAL |+>)
# ============================================================

def tfim_window_ansatz(n_qubits, window=5, layers=1):

    if window > n_qubits:
        window = n_qubits

    qc = QuantumCircuit(n_qubits)

    # ⭐ initial state |+> (molto importante per TFIM)
    #for q in range(n_qubits):
        #qc.h(q)

    gamma = ParameterVector("γ", layers)
    beta = ParameterVector("β", layers)

    starts = range(0, n_qubits - window + 1)

    for l in range(layers):
        for s in starts:
            qs = list(range(s, s + window))

            # ZZ interactions locali
            for i in range(len(qs) - 1):
                a, b = qs[i], qs[i + 1]
                qc.cx(a, b)
                qc.rz(2 * gamma[l], b)
                qc.cx(a, b)

            # transverse field
            for q in qs:
                qc.rx(2 * beta[l], q)

    return qc


# ============================================================
# MULTI-START VQE (Powell optimizer)
# ============================================================

def vqe_energy(ansatz, H, maxiter=1500, n_starts=5, seed=123):

    estimator = Estimator()
    rng = np.random.default_rng(seed)

    def objective(x):
        job = estimator.run(ansatz, H, parameter_values=x)
        return job.result().values[0]

    best_E = float("inf")
    best_params = None

    print(f"\nRunning VQE ({n_starts} random starts, Powell optimizer)...")

    for k in range(n_starts):

        # ⭐ inizializzazione vicino a 0 (meglio per TFIM)
        x0 = rng.normal(0, 0.3, len(ansatz.parameters))

        res = minimize(
            objective,
            x0,
            method="Powell",
            options={"maxiter": maxiter}
        )

        print(f"Start {k+1}/{n_starts} → energy {res.fun:.6f}")

        if res.fun < best_E:
            best_E = res.fun
            best_params = res.x

    return best_E, best_params


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    n = 6
    J = 1.0
    h = 0.7

    print("\n==============================")
    print("TFIM WINDOW ANSATZ TEST (OPTIMIZED)")
    print("==============================")

    # --- Hamiltonian ---
    print("\nBuilding TFIM Hamiltonian...")
    H = tfim_operator(n, J, h)

    # --- Exact energy ---
    print("\nComputing exact ground energy...")
    E_exact = exact_ground_energy(H)
    print("Exact ground energy:", E_exact)

    # --- Ansatz ---
    print("\nBuilding window ansatz...")
    ansatz = tfim_window_ansatz(n, window=5, layers=1)

  
    # --- VQE ---
    E_vqe, best_params = vqe_energy(ansatz, H)
    
    param_dict = dict(zip(ansatz.parameters, best_params))

    opt_circuit = ansatz.assign_parameters(param_dict)
    
    
    
    
    # ============================================================
    # TEST HARDWARE LAGRANGE FIT
    # ============================================================

    print("\n==============================")
    print("TESTING LAGRANGE FIT")
    print("==============================")

    from fit_lagrange import fit_for_lagrange

    reduced = fit_for_lagrange(ansatz)
    if reduced is not None and reduced.parameters:
        reduced = reduced.assign_parameters(param_dict)

    
    if reduced is None:
        print("\n❌ Circuit cannot be fitted to 5 qubits")
    else:
        print("\n✅ Circuit fits Lagrange hardware!")

        print("\nOriginal qubits:", ansatz.num_qubits)
        print("Reduced qubits:", reduced.num_qubits)

        print("\nOriginal depth:", ansatz.depth())
        print("Reduced depth:", reduced.depth())



    print("\nGate counts original:", ansatz.count_ops())
    if reduced:
        print("Gate counts reduced:", reduced.count_ops())

        print("Number of parameters:", len(ansatz.parameters))
        print("Circuit depth:", ansatz.depth())














    # --- Results ---
    print("\n==============================")
    print("RESULTS")
    print("==============================")

    print("Exact energy:", E_exact)
    print("Best VQE energy:", E_vqe)
    print("Absolute error:", abs(E_vqe - E_exact))
    print("Relative error:", abs(E_vqe - E_exact) / abs(E_exact))


    # --- TEST FOR LAGRANGE CIRCUIT simulation ---

    print("\nLogical VQE energy:", E_vqe)
    
    
    # ============================================================
    # DEPLOY TEST: LOGICAL vs COMPRESSED (ZZ-ONLY ENERGY)
    # ============================================================



    print("\n==============================")
    print("DEPLOY TEST (ZZ part only)")
    print("==============================")

    # 1️ Parametri ottimali già trovati prima:
    # E_vqe, best_params = vqe_energy(ansatz, H)
    # Assumo che best_params sia disponibile

    # Ricostruisci circuito ottimo
    print("Reduced circuit parameters:", reduced.parameters)

    
    param_dict = dict(zip(ansatz.parameters, best_params))
    opt_circuit = ansatz.assign_parameters(param_dict)

    # Aggiungi misura completa al circuito logico
    opt_meas = opt_circuit.copy()
    opt_meas.measure_all()


    shots = 20000
    sim = AerSimulator(seed_simulator=1234)

    # Esegui circuito logico
    job_logical = sim.run(opt_meas, shots=shots)
    counts_logical = job_logical.result().get_counts()

    # Esegui circuito compresso
    job_reduced = sim.run(reduced, shots=shots)
    counts_reduced = job_reduced.result().get_counts()

# ============================================================
# Funzioni per calcolo ⟨ZZ⟩
# ============================================================

def exp_ZZ_from_counts(counts, i, j, n):
    shots = sum(counts.values())
    s = 0.0
    for bitstring, c in counts.items():
        # Qiskit: bitstring[0] = bit classico più significativo (n-1)
        bi = int(bitstring[n - 1 - i])
        bj = int(bitstring[n - 1 - j])
        zi = +1 if bi == 0 else -1
        zj = +1 if bj == 0 else -1
        s += (zi * zj) * c
    return s / shots

def energy_tfim_zz_only(counts, n, J, boundary="open"):
    e_zz = 0.0
    max_edge = n - 1 if boundary == "open" else n
    for i in range(max_edge):
        j = (i + 1) % n
        e_zz += exp_ZZ_from_counts(counts, i, j, n)
    return -J * e_zz

# ============================================================
# Confronto energia ZZ
# ============================================================

Ezz_logical = energy_tfim_zz_only(counts_logical, n=6, J=J)
Ezz_deploy  = energy_tfim_zz_only(counts_reduced,  n=6, J=J)

print("\nZZ-only energy (logical):", Ezz_logical)
print("ZZ-only energy (deploy): ", Ezz_deploy)
print("Difference:", abs(Ezz_logical - Ezz_deploy))

print("\n==============================")
print("X TERM (logical) + TOTAL ENERGY DECOMPOSITION")
print("==============================")

# --- prepara circuito per misurare X: H su tutti + misura
opt_meas_x = opt_circuit.copy()
for q in range(opt_circuit.num_qubits):
    opt_meas_x.h(q)
opt_meas_x.measure_all()

# esegui
job_x = sim.run(opt_meas_x, shots=shots)
counts_x = job_x.result().get_counts()

def exp_Z_from_counts(counts, qubit_index, n):
    """⟨Z_i⟩ from Z-basis counts."""
    shots_ = sum(counts.values())
    s = 0.0
    for bitstring, c in counts.items():
        b = int(bitstring[n - 1 - qubit_index])
        z = +1 if b == 0 else -1
        s += z * c
    return s / shots_

def energy_tfim_x_only(counts_x, n, h):
    """
    counts_x proviene da misura dopo H su tutti i qubit.
    Quindi ⟨X_i⟩ = ⟨Z_i⟩_counts_x.
    """
    ex_sum = 0.0
    for i in range(n):
        ex_sum += exp_Z_from_counts(counts_x, i, n)
    return -h * ex_sum

# calcola E_X e totale (da sampling)
E_x_logical = energy_tfim_x_only(counts_x, n=6, h=h)
E_total_sampling_logical = Ezz_logical + E_x_logical

print("E_ZZ (logical, sampling):", Ezz_logical)
print("E_X  (logical, sampling):", E_x_logical)
print("E_total (logical, sampling):", E_total_sampling_logical)

print("\nCheck vs Estimator VQE energy (logical):", E_vqe)
print("Abs diff (Estimator vs sampling total):", abs(E_vqe - E_total_sampling_logical))
