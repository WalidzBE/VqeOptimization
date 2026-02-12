# Features

## Hamiltonian Framework
The library provides a registry-driven framework for building Hamiltonians from typed specifications. Each
`HamiltonianSpec` validates inputs before operator construction, enabling consistent CLI integration and reproducible
configuration.

## Transverse-Field Ising Model (TFIM)
The TFIM Hamiltonian implemented here follows:

$$
H = -J \sum_{\langle i, j \rangle} Z_i Z_j - h \sum_i X_i
$$

Supported options:
- Open or periodic boundary conditions.
- Arbitrary number of qubits (subject to exact diagonalization limits for scans).
- EfficientSU2 as the default ansatz for VQE runs.

## Molecular Hamiltonian (H2O)
The chemistry pipeline constructs an H2O Hamiltonian using Qiskit Nature and PySCF. It supports:
- Active-space reduction.
- Freeze-core transformation.
- Jordan-Wigner or parity mapping.
- Optional Z2 symmetry tapering when using parity mapping.

## VQE Runners
Command-line runners cover common workflows:
- `run_simulation`: single VQE optimization with Aer or IQM estimator.
- `run_benchmark`: repeated runs with circuit transpilation metrics.
- `run_tfim_scan`: parameter sweeps over $h/J$ with optional exact diagonalization.

## Backend Support
- Local simulation via Qiskit Aer Estimator.
- Optional IQM backend integration for execution on IQM hardware backends.
