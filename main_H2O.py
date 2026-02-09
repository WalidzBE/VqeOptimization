# ============================================================
# STEP 0 — IMPORT LIBRARIES
# We use Qiskit Nature to build a quantum chemistry problem
# and map it to a qubit Hamiltonian.
# ============================================================

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import FreezeCoreTransformer, ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit.quantum_info.analysis import Z2Symmetries


# ============================================================
# STEP 1 — DEFINE THE MOLECULE (H2O)
# Geometry in Angstrom. This defines the physical system.
# ============================================================

driver = PySCFDriver(
    atom="O 0.0 0.0 0.0; H 0.0 0.0 0.958; H 0.9266 0.0 -0.2396",
    basis="sto3g",          # Minimal basis → small number of qubits
    charge=0,               # Neutral molecule
    spin=0,                 # Singlet ground state
    unit=DistanceUnit.ANGSTROM,
)

problem = driver.run()


# ============================================================
# STEP 2 — FREEZE CORE ORBITALS
# Core electrons do not contribute much to bonding,
# so we remove them to reduce the quantum problem size.
# ============================================================

problem_fc = FreezeCoreTransformer().transform(problem)


# ============================================================
# STEP 3 — DEFINE THE ACTIVE SPACE
# We keep only a subset of electrons and orbitals that are
# most relevant for correlation and bonding.
# This is a key chemistry → hardware reduction step.
# ============================================================

active_space = ActiveSpaceTransformer(
    num_electrons=6,          # Number of active electrons
    num_spatial_orbitals=5    # Number of active spatial orbitals
)

problem_as = active_space.transform(problem_fc)


# ============================================================
# STEP 4 — MAP THE ELECTRONIC HAMILTONIAN TO QUBITS
# We first build the second-quantized Hamiltonian (fermions)
# and then map it to qubits using Jordan–Wigner mapping.
# ============================================================

jw_mapper = JordanWignerMapper()
second_q_op = problem_as.hamiltonian.second_q_op()
qubit_hamiltonian_jw = jw_mapper.map(second_q_op)

print("\n===== AFTER JORDAN–WIGNER MAPPING =====")
print("Number of qubits required:", qubit_hamiltonian_jw.num_qubits)
print("Number of Pauli terms:", len(qubit_hamiltonian_jw))


# ============================================================
# STEP 5 — PARITY MAPPING (BETTER FOR TAPERING)
# Parity mapping exposes Z2 symmetries that allow qubit reduction.
# ============================================================

parity_mapper = ParityMapper()
qubit_op = parity_mapper.map(second_q_op)

print("\n===== BEFORE TAPERING (PARITY MAPPING) =====")
print("Number of qubits:", qubit_op.num_qubits)


# ============================================================
# STEP 6 — FIND Z2 SYMMETRIES
# These symmetries come from physical conservation laws
# (particle number, spin parity, etc.).
# Each symmetry can potentially remove one qubit.
# ============================================================

z2_symmetries = Z2Symmetries.find_z2_symmetries(qubit_op)

print("\n===== Z2 SYMMETRIES FOUND =====")
print("Number of symmetries:", len(z2_symmetries.symmetries))


# ============================================================
# STEP 7 — TAPER THE HAMILTONIAN
# Tapering removes qubits whose values are fixed by symmetry.
# The result is a list of possible symmetry sectors.
# We select one sector (first) for now.
# ============================================================

tapered_ops = z2_symmetries.taper(qubit_op)

print("\nNumber of tapered operators (symmetry sectors):", len(tapered_ops))

tapered_op = tapered_ops[0]  # choose first symmetry sector

print("\n===== AFTER TAPERING =====")
print("Number of qubits:", tapered_op.num_qubits)
print("Number of Pauli terms:", len(tapered_op))


# ============================================================
# STEP 8 — MEASUREMENT GROUPING (CORE OPTIMIZATION PART)
# We group Pauli terms that can be measured in the same basis
# to reduce the number of circuits.
# ============================================================

from qiskit.quantum_info import SparsePauliOp

print("\n===== MEASUREMENT GROUPING =====")

# Convert tapered Hamiltonian to SparsePauliOp
sparse_pauli = SparsePauliOp(tapered_op)

print("Original number of Pauli terms:", len(sparse_pauli))

# Group terms that commute qubit-wise
groups = sparse_pauli.group_commuting(qubit_wise=True)

print("Number of commuting groups:", len(groups))
print("Reduction factor:", len(sparse_pauli) / len(groups))


# ============================================================
# STEP 9 — BUILD A HARDWARE-EFFICIENT ANSATZ (7 QUBITS)
# This is the parametrized circuit used by VQE to prepare |psi(theta)>
# ============================================================

from qiskit.circuit.library import TwoLocal
from qiskit import QuantumCircuit

num_qubits = tapered_op.num_qubits  # should be 7

# TwoLocal = common hardware-efficient ansatz
ansatz = TwoLocal(
    num_qubits=num_qubits,
    rotation_blocks=['ry', 'rz'],     # single-qubit rotations
    entanglement_blocks='cx',         # entangling gates
    entanglement='linear',            # chain connectivity
    reps=2,                           # number of layers (can tune)
    insert_barriers=True
)

print("\n===== ANSATZ INFO =====")
print("Number of qubits:", ansatz.num_qubits)
print("Number of parameters:", ansatz.num_parameters)
print("Circuit depth (untranspiled):", ansatz.depth())


# ============================================================
# STEP 10 — VISUALIZE THE ANSATZ CIRCUIT
# ============================================================

print("\n===== DECOMPOSED ANSATZ CIRCUIT =====")
print(ansatz.decompose().draw('text'))


print("\n===== ANSATZ METRICS =====")
print("Circuit depth:", ansatz.decompose().depth())
print("Number of CNOTs:", ansatz.decompose().count_ops().get('cx', 0))
print("Total gates:", ansatz.decompose().size())


# ============================================================
# EXACT GROUND STATE ENERGY (CLASSICAL NUMPY)
# ============================================================

import numpy as np

print("\n===== EXACT DIAGONALIZATION (CLASSICAL) =====")

# Convert SparsePauliOp → dense matrix
H_matrix = tapered_op.to_matrix()

# Diagonalize
eigenvalues, _ = np.linalg.eigh(H_matrix)

exact_energy = np.min(eigenvalues).real

print("Exact ground state energy:", exact_energy)

# ============================================================
# STEP 11 — RUN BASELINE VQE (IDEAL SIMULATOR, QISKIT 2.3)
# ============================================================

from qiskit_aer.primitives import Estimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SPSA

print("\n===== RUNNING BASELINE VQE (IDEAL SIMULATION) =====")

# Aer Estimator = statevector simulation (noiseless)
estimator = Estimator()

# Optimizer
#optimizer = SPSA(maxiter=2)

from qiskit_algorithms.optimizers import COBYLA

optimizer = COBYLA(maxiter=5)


# Build VQE
vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)

# Run VQE
result = vqe.compute_minimum_eigenvalue(operator=tapered_op)

print("\n===== VQE RESULTS =====")
print("Estimated ground state energy:", result.eigenvalue.real)
print("Number of optimizer evaluations:", result.optimizer_evals)
