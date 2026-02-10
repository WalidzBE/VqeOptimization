# VQE Optimization (TFIM)

Modular Transverse-Field Ising Model (TFIM) Hamiltonian and VQE runners using the latest Qiskit primitives.

## Quickstart (uv)

Create a Python 3.10–3.12 environment and install dependencies:

```bash
uv python install 3.12
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run a VQE simulation (Aer estimator):

```bash
python -m runners.run_simulation \
  --hamiltonian tfim \
  --n_qubits 8 --J 1.0 --h 0.7 --boundary open \
  --ansatz efficient_su2 --reps 2 \
  --optimizer cobyla --maxiter 200 \
  --shots 4000 \
  --seed 123
```

Run chemistry (H2O) with parity mapping + tapering:

```bash
python -m runners.run_simulation \
  --hamiltonian chemistry \
  --molecule h2o \
  --basis sto3g \
  --active_electrons 6 \
  --active_orbitals 5 \
  --mapper parity \
  --taper \
  --taper_sector 0 \
  --ansatz two_local --reps 2 \
  --optimizer cobyla --maxiter 200 \
  --shots 4000 \
  --seed 123
```

Scan TFIM energy vs h/J:

```bash
python -m runners.run_tfim_scan \
  --n_qubits 8 --J 1.0 --boundary open \
  --ratio_start 0.2 --ratio_end 1.8 --num_points 17 \
  --ansatz efficient_su2 --reps 2 \
  --optimizer cobyla --maxiter 200 \
  --shots 4000 \
  --seed 123 \
  --output tfim_scan.png
```

Exact diagonalization notes:
- The scan uses exact diagonalization by default and will error for large `n_qubits`.
- Use `--no-exact` to skip exact energies or `--exact_max_qubits` to raise the limit.

Run a benchmark (repeated runs + circuit metrics):

```bash
python -m runners.run_benchmark \
  --hamiltonian tfim \
  --n_qubits 8 --J 1.0 --h 0.7 --boundary open \
  --ansatz efficient_su2 --reps 2 \
  --optimizer cobyla --maxiter 200 \
  --shots 4000 \
  --seed 123 \
  --runs 3
```

Run on hardware (inject backend):

```bash
export QISKIT_IBM_TOKEN="your_token"
python -m runners.run_hardware \
  --hamiltonian tfim \
  --n_qubits 8 --J 1.0 --h 0.7 --boundary open \
  --ansatz efficient_su2 --reps 2 \
  --optimizer cobyla --maxiter 200 \
  --shots 4000 \
  --seed 123 \
  --backend_name ibm_nairobi \
  --ibm_channel ibm_quantum \
  --ibm_instance your_hub/group/project
```

Notes:
- `QISKIT_IBM_TOKEN` is read by `QiskitRuntimeService` (no secrets in code).
- If you do not pass `--shots`, the estimator uses its default (statevector for Aer in most configs).

## References

- https://www.arxiv.org/pdf/2601.17515

## Scripts and parameters

Available scripts:
- `python -m runners.run_simulation`
- `python -m runners.run_benchmark`
- `python -m runners.run_hardware`
- `python -m runners.run_tfim_scan`

Common parameters:
- `--hamiltonian` (default `tfim`)
- `--ansatz` (`efficient_su2`, `two_local`)
- `--reps`
- `--optimizer` (`cobyla`, `spsa`)
- `--maxiter`
- `--shots`
- `--seed`

TFIM parameters:
- `--n_qubits`
- `--J`
- `--h`
- `--boundary` (`open`, `periodic`)

TFIM scan parameters:
- `--n_qubits`
- `--J`
- `--boundary` (`open`, `periodic`)
- `--ratio_start`
- `--ratio_end`
- `--num_points`
- `--exact` / `--no-exact`
- `--exact_max_qubits`
- `--output`
- `--emit_json`

Chemistry parameters (H2O supported):
- `--molecule` (`h2o`)
- `--basis`
- `--charge`
- `--spin`
- `--unit` (`angstrom`)
- `--freeze_core` / `--no-freeze_core`
- `--active_electrons`
- `--active_orbitals`
- `--mapper` (`jw`, `parity`)
- `--taper` / `--no-taper`
- `--taper_sector`

## Project layout

```
.
├── pyproject.toml
├── README.md
├── hamiltonians/
│   ├── __init__.py
│   ├── base.py
│   └── tfim.py
├── runners/
│   ├── __init__.py
│   ├── common.py
│   ├── run_simulation.py
│   ├── run_hardware.py
│   └── run_benchmark.py
└── tests/
    └── test_tfim.py
```
