from __future__ import annotations

import json
import sys
from typing import Any

import argparse

from hamiltonians.base import add_hamiltonian_args, get_hamiltonian
from runners.common import (
    build_ansatz,
    build_base_parser,
    build_optimizer,
    build_spec_and_operator,
    result_payload,
    run_vqe,
)


def build_runtime_estimator(
    backend_name: str,
    shots: int | None,
    seed: int | None,
    channel: str | None,
    instance: str | None,
) -> tuple[Any, Any]:
    from qiskit_ibm_runtime import Estimator as RuntimeEstimator
    from qiskit_ibm_runtime import QiskitRuntimeService, Session

    service = QiskitRuntimeService(channel=channel, instance=instance)

    options = None
    try:
        from qiskit_ibm_runtime import Options

        options = Options()
        if shots is not None:
            options.execution.shots = shots
        if seed is not None and getattr(options, "simulator", None) is not None:
            options.simulator.seed_simulator = seed
    except Exception:
        options = None

    session = Session(service=service, backend=backend_name)
    estimator = RuntimeEstimator(session=session, options=options) if options else RuntimeEstimator(session=session)
    return estimator, session


def parse_args(argv: list[str]) -> Any:
    base = build_base_parser("VQE hardware runner", add_hamiltonian=True)
    pre_args, _ = base.parse_known_args(argv)
    ham = get_hamiltonian(pre_args.hamiltonian)

    parser = argparse.ArgumentParser(description="VQE hardware runner", parents=[base], allow_abbrev=False)
    parser.add_argument("--backend_name", default=None)
    parser.add_argument("--ibm_channel", default=None)
    parser.add_argument("--ibm_instance", default=None)
    add_hamiltonian_args(parser, ham)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.backend_name:
        raise SystemExit("--backend_name is required for hardware runs")

    operator, n_qubits = build_spec_and_operator(args)
    ansatz = build_ansatz(args.ansatz, n_qubits, args.reps)
    optimizer = build_optimizer(args.optimizer, args.maxiter, args.seed)

    estimator, session = build_runtime_estimator(
        backend_name=args.backend_name,
        shots=args.shots,
        seed=args.seed,
        channel=args.ibm_channel,
        instance=args.ibm_instance,
    )

    try:
        result, meta = run_vqe(operator, ansatz, optimizer, estimator, args.seed)
    finally:
        session.close()

    payload = result_payload(result, meta)
    payload["backend"] = args.backend_name
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
