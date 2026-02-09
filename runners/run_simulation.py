from __future__ import annotations

import json
import sys

from qiskit_aer.primitives import Estimator as AerEstimator

from runners.common import (
    build_ansatz,
    build_optimizer,
    build_parser_with_hamiltonian,
    build_spec_and_operator,
    result_payload,
    run_vqe,
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser_with_hamiltonian("VQE simulation runner", argv or sys.argv[1:])
    operator, n_qubits = build_spec_and_operator(args)

    ansatz = build_ansatz(args.ansatz, n_qubits, args.reps)
    optimizer = build_optimizer(args.optimizer, args.maxiter, args.seed)

    run_options = {}
    if args.shots is not None:
        run_options["shots"] = args.shots
    if args.seed is not None:
        run_options["seed_simulator"] = args.seed

    estimator = AerEstimator(run_options=run_options) if run_options else AerEstimator()

    result, meta = run_vqe(operator, ansatz, optimizer, estimator, args.seed)
    payload = result_payload(result, meta)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
