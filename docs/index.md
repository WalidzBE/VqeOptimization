# VQE Optimization

## Badges
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://WalidzBE.github.io/VqeOptimization/)
[![Build](https://img.shields.io/github/actions/workflow/status/walidzbe/VqeOptimization/docs.yml?branch=main&label=build)](https://github.com/walidzbe/VqeOptimization/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview
VQE Optimization provides modular Hamiltonian builders and command-line runners for Variational Quantum Eigensolver (VQE)
workflows. You can run any Hamiltonian by adding a small module under `hamiltonians/`, without touching low-level quantum
plumbing. The current bundle includes TFIM and an H2O Hamiltonian using Qiskit primitives, with optional IQM backend support.

## Project Scope
- Hamiltonian registry with typed specifications and validation.
- TFIM Hamiltonian construction with open or periodic boundary conditions.
- Molecular Hamiltonian construction for H2O with active-space reduction and optional tapering.
- Auto-generated CLI flags from Hamiltonian specs.
- CLI runners for single VQE simulations, benchmarks, and TFIM scans.
- Estimator backends: Qiskit Aer simulator or IQM hardware (with optional naive MOVE transpilation).

## Documentation Map
- Features: capabilities and model details.
- Installation: environment setup with `uv`.
- Quickstart: minimal working examples.
- Examples: task-focused usage patterns.
- API Reference: auto-generated from public Python modules.

## License
See the repository for license details.
