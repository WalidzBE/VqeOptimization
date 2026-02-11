from __future__ import annotations

import argparse
import json
import logging
import sys

from qiskit_aer.primitives import Estimator as AerEstimator

from hamiltonians.base import add_hamiltonian_args, get_hamiltonian
from runners.common import (
    add_iqm_args,
    build_ansatz,
    build_base_parser,
    build_iqm_estimator,
    build_optimizer,
    build_spec_and_operator,
    result_payload,
    run_vqe,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    base = build_base_parser("VQE simulation runner", add_hamiltonian=True)
    add_iqm_args(base)
    pre_args, _ = base.parse_known_args(argv)
    ham = get_hamiltonian(pre_args.hamiltonian)

    parser = argparse.ArgumentParser(description="VQE simulation runner", parents=[base], allow_abbrev=False)
    add_hamiltonian_args(parser, ham)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("iqm").setLevel(logging.WARNING)
    logging.getLogger("qiskit_aer").setLevel(logging.WARNING)
    args = parse_args(argv or sys.argv[1:])
    if args.estimator != "iqm" and any(
        [args.iqm_url, args.iqm_backend, args.iqm_tokens_file, args.iqm_naive_move]
    ):
        raise SystemExit("--iqm_* options require --estimator iqm")
    operator, n_qubits = build_spec_and_operator(args)

    ansatz = build_ansatz(args.ansatz, n_qubits, args.reps)
    optimizer = build_optimizer(args.optimizer, args.maxiter, args.seed)

    aer_run_options = {}
    if args.shots is not None:
        aer_run_options["shots"] = args.shots
    if args.seed is not None:
        aer_run_options["seed_simulator"] = args.seed

    if args.estimator == "aer":
        estimator = AerEstimator(run_options=aer_run_options) if aer_run_options else AerEstimator()
        iqm_meta = {}
    else:
        estimator, ansatz, iqm_meta = build_iqm_estimator(
            iqm_url=args.iqm_url,
            iqm_backend=args.iqm_backend,
            iqm_tokens_file=args.iqm_tokens_file,
            naive_move=args.iqm_naive_move,
            shots=args.shots,
            circuit=ansatz,
        )

    result, meta = run_vqe(operator, ansatz, optimizer, estimator, args.seed)
    meta["estimator"] = args.estimator
    meta.update(iqm_meta)
    payload = result_payload(result, meta)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
