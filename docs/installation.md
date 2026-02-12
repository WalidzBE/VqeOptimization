# Installation

## Requirements
- Python 3.11 to 3.12
- `uv` for environment and dependency management

## Install With `uv`
```bash
uv python install 3.12
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

The project depends on Qiskit, Qiskit Aer, Qiskit Algorithms, Qiskit Nature, PySCF, NumPy, and Matplotlib. These are
installed automatically by the command above.

## Verify Installation
```bash
python -m runners.run_simulation --help
```
