# Features

## Hamiltonian Framework
The library provides a registry-driven framework for building Hamiltonians from typed specifications. Add a new module in
`hamiltonians/` with a spec + `build_operator`, register it, and it becomes runnable—no need to handle quantum-level
configuration details. Each `HamiltonianSpec` validates inputs before operator construction, enabling consistent CLI
integration and reproducible configuration.

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

## CLI From Specs
Hamiltonian specs are typed dataclasses; the CLI flags for each Hamiltonian are auto-generated from those fields. Adding a new Hamiltonian module and registering it exposes its parameters automatically via `--<field>` flags.

## VQE Runners
Command-line runners cover common workflows:
- `run_simulation`: single VQE optimization with Aer or IQM estimator.
- `run_benchmark`: repeated runs with circuit transpilation metrics.
- `run_tfim_scan`: parameter sweeps over $h/J$ with optional exact diagonalization.

## Ansatz and Optimizer Catalog
- Ansatz choices: `efficient_su2` (hardware-efficient SU2 with linear entanglement) and `two_local` (`ry/rz` rotations + `cx` linear entanglement).
- Optimizer choices: `cobyla` (deterministic, gradient-free) and `spsa` (stochastic, shot-noise-friendly).

## Backend Support
- Local simulation via Qiskit Aer Estimator.
- Optional IQM backend integration for execution on IQM hardware backends.
- IQM-specific options include URL/backend selection, token file path, and an optional `--iqm_naive_move` transpilation pass to match device MOVE gates.
