from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import matplotlib
import numpy as np
from qiskit_aer.primitives import Estimator as AerEstimator

from hamiltonians.tfim import TFIMSpec, build_operator
from runners.common import build_ansatz, build_optimizer, result_payload, run_vqe


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args(argv: list[str]) -> Any:
    parser = argparse.ArgumentParser(description="TFIM VQE scan", allow_abbrev=False)
    parser.add_argument("--n_qubits", type=int, required=True)
    parser.add_argument("--J", type=float, required=True)
    parser.add_argument("--boundary", default="open", choices=["open", "periodic"])
    parser.add_argument("--ratio_start", type=float, default=0.2)
    parser.add_argument("--ratio_end", type=float, default=1.8)
    parser.add_argument("--num_points", type=int, default=17)
    parser.add_argument("--ansatz", default="efficient_su2", choices=["efficient_su2", "two_local"])
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--optimizer", default="cobyla", choices=["cobyla", "spsa"])
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exact", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exact_max_qubits", type=int, default=12)
    parser.add_argument("--output", default="tfim_scan.png")
    parser.add_argument("--emit_json", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.J == 0.0:
        raise SystemExit("J must be non-zero to scan h/J ratios")
    if args.num_points < 2:
        raise SystemExit("num_points must be >= 2")
    if args.exact and args.n_qubits > args.exact_max_qubits:
        raise SystemExit(
            f"Exact diagonalization scales as 2^n; set --no-exact or raise --exact_max_qubits "
            f"(current n_qubits={args.n_qubits})"
        )

    ratios = np.linspace(args.ratio_start, args.ratio_end, args.num_points)

    ansatz = build_ansatz(args.ansatz, args.n_qubits, args.reps)
    optimizer = build_optimizer(args.optimizer, args.maxiter, args.seed)

    run_options = {}
    if args.shots is not None:
        run_options["shots"] = args.shots
    if args.seed is not None:
        run_options["seed_simulator"] = args.seed

    estimator = AerEstimator(run_options=run_options) if run_options else AerEstimator()

    energies = []
    exact_energies = []
    details = []

    for ratio in ratios:
        h_value = float(ratio * args.J)
        spec = TFIMSpec(n_qubits=args.n_qubits, J=args.J, h=h_value, boundary=args.boundary)
        operator = build_operator(spec)
        result, meta = run_vqe(operator, ansatz, optimizer, estimator, args.seed)
        payload = result_payload(result, meta)
        energies.append(payload["energy"])
        exact_energy = None
        if args.exact:
            exact_energy = float(np.linalg.eigvalsh(operator.to_matrix())[0].real)
            exact_energies.append(exact_energy)
        details.append({"h_over_j": float(ratio), "h": h_value, "exact_energy": exact_energy, **payload})

    plt.figure(figsize=(7, 4))
    plt.plot(ratios, energies, marker="o", label="VQE")
    if args.exact:
        plt.plot(ratios, exact_energies, marker="s", label="Exact")
    plt.xlabel("h/J")
    plt.ylabel("Estimated ground energy")
    plt.title(f"TFIM VQE scan (n={args.n_qubits}, J={args.J}, boundary={args.boundary})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)

    print(json.dumps({"output": args.output, "points": details}, indent=2, sort_keys=True))
    if args.emit_json:
        with open(args.emit_json, "w", encoding="utf-8") as handle:
            json.dump({"output": args.output, "points": details}, handle, indent=2, sort_keys=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
