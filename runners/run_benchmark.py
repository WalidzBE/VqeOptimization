from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Any

from qiskit_aer.primitives import Estimator as AerEstimator

from hamiltonians.base import add_hamiltonian_args, get_hamiltonian
from runners.common import (
    build_ansatz,
    build_base_parser,
    build_optimizer,
    build_spec_and_operator,
    result_payload,
    run_vqe,
    transpile_metrics,
)


def parse_args(argv: list[str]) -> Any:
    base = build_base_parser("VQE benchmark runner", add_hamiltonian=True)
    pre_args, _ = base.parse_known_args(argv)
    ham = get_hamiltonian(pre_args.hamiltonian)

    parser = argparse.ArgumentParser(description="VQE benchmark runner", parents=[base], allow_abbrev=False)
    parser.add_argument("--runs", type=int, default=3)
    add_hamiltonian_args(parser, ham)
    return parser.parse_args(argv)


def bind_optimal_params(ansatz, result: Any):
    if getattr(result, "optimal_parameters", None):
        return ansatz.assign_parameters(result.optimal_parameters)
    optimal_point = getattr(result, "optimal_point", None)
    if optimal_point is not None:
        return ansatz.assign_parameters(dict(zip(ansatz.parameters, optimal_point)))
    return ansatz


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    operator, n_qubits = build_spec_and_operator(args)

    ansatz = build_ansatz(args.ansatz, n_qubits, args.reps)
    optimizer = build_optimizer(args.optimizer, args.maxiter, args.seed)

    run_options = {}
    if args.shots is not None:
        run_options["shots"] = args.shots
    if args.seed is not None:
        run_options["seed_simulator"] = args.seed

    estimator = AerEstimator(run_options=run_options) if run_options else AerEstimator()

    energies = []
    evals = []
    run_times = []
    metrics = []

    for _ in range(args.runs):
        result, meta = run_vqe(operator, ansatz, optimizer, estimator, args.seed)
        payload = result_payload(result, meta)
        energies.append(payload["energy"])
        evals.append(payload["optimizer_evals"])
        run_times.append(meta["total_time_s"])

        bound = bind_optimal_params(ansatz, result)
        metrics.append(transpile_metrics(bound, args.seed))

    shots_cost = None
    if args.shots is not None and all(isinstance(e, int) for e in evals):
        shots_cost = args.shots * int(statistics.mean(evals)) * operator.size

    report = {
        "runs": args.runs,
        "energy_mean": statistics.mean(energies) if energies else None,
        "energy_std": statistics.pstdev(energies) if len(energies) > 1 else 0.0,
        "wall_time_mean_s": statistics.mean(run_times) if run_times else None,
        "optimizer_evals_mean": statistics.mean(evals) if evals else None,
        "shots_cost_estimate": shots_cost,
        "transpile": metrics,
        "energies": energies,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
