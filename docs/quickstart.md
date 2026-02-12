# Quickstart

## Minimal Working Example
Run a single TFIM VQE simulation using the Aer estimator:

```bash
python -m runners.run_simulation \
  --hamiltonian tfim \
  --n_qubits 8 --J 1.0 --h 0.7 --boundary open \
  --ansatz efficient_su2 --reps 2 \
  --optimizer cobyla --maxiter 200 \
  --shots 4000 \
  --seed 123
```

The command prints a JSON payload with the estimated ground energy, optimal parameters, and metadata.

## TFIM Scan
Scan the TFIM ground-state energy as a function of $h/J$:

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

Notes:
- Exact diagonalization is enabled by default and scales as $2^n$.
- Use `--no-exact` to disable exact energies or `--exact_max_qubits` to raise the limit.

## Chemistry Example (H2O)
Run VQE for the H2O Hamiltonian using parity mapping and tapering:

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
