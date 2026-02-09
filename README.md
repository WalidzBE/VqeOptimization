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

Run tests:

```bash
pytest
```

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
