# Examples Overview

This page collects representative workflows using the provided CLI runners. Each example mirrors the arguments
validated in code.

## TFIM Scan With Exact Diagonalization
```bash
python -m runners.run_tfim_scan \
  --n_qubits 6 --J 1.0 --boundary open \
  --ratio_start 0.4 --ratio_end 1.6 --num_points 9 \
  --ansatz efficient_su2 --reps 2 \
  --optimizer cobyla --maxiter 200 \
  --shots 2000 \
  --seed 123 \
  --output tfim_scan.png
```

## TFIM Scan Without Exact Diagonalization
```bash
python -m runners.run_tfim_scan \
  --n_qubits 12 --J 1.0 --boundary periodic \
  --ratio_start 0.4 --ratio_end 1.6 --num_points 9 \
  --ansatz efficient_su2 --reps 2 \
  --optimizer cobyla --maxiter 200 \
  --shots 2000 \
  --seed 123 \
  --no-exact \
  --output tfim_scan_periodic.png
```

## VQE Benchmark
Run repeated VQE optimizations and collect transpilation metrics:

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

## IQM Estimator Example
If you have IQM access, use the IQM estimator with tokens stored in a file:

```bash
python -m runners.run_simulation \
  --hamiltonian tfim \
  --n_qubits 4 --J 1.0 --h 0.7 --boundary open \
  --ansatz efficient_su2 --reps 2 \
  --optimizer cobyla --maxiter 200 \
  --shots 2048 \
  --seed 123 \
  --estimator iqm \
  --iqm_tokens_file ./tokens.json
```
