from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any

import matplotlib
import numpy as np
from qiskit_aer.primitives import Estimator as AerEstimator

from hamiltonians.tfim import TFIMSpec, build_operator
from runners.common import add_iqm_args, build_ansatz, build_iqm_estimator, build_optimizer, result_payload, run_vqe


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)


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
    add_iqm_args(parser)
    parser.add_argument("--exact", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exact_max_qubits", type=int, default=12)
    parser.add_argument("--output", default="tfim_scan.png")
    parser.add_argument("--emit_json", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("iqm").setLevel(logging.WARNING)
    logging.getLogger("qiskit_aer").setLevel(logging.WARNING)
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
    if args.estimator != "iqm" and any(
        [args.iqm_url, args.iqm_backend, args.iqm_tokens_file, args.iqm_naive_move]
    ):
        raise SystemExit("--iqm_* options require --estimator iqm")

    ratios = np.linspace(args.ratio_start, args.ratio_end, args.num_points)

    ansatz = build_ansatz(args.ansatz, args.n_qubits, args.reps)
    optimizer = build_optimizer(args.optimizer, args.maxiter, args.seed)

    aer_run_options = {}
    if args.shots is not None:
        aer_run_options["shots"] = args.shots
    if args.seed is not None:
        aer_run_options["seed_simulator"] = args.seed

    meta_out = {"estimator": args.estimator}

    if args.estimator == "aer":
        estimator = AerEstimator(run_options=aer_run_options) if aer_run_options else AerEstimator()
    else:
        estimator, ansatz, iqm_meta = build_iqm_estimator(
            iqm_url=args.iqm_url,
            iqm_backend=args.iqm_backend,
            iqm_tokens_file=args.iqm_tokens_file,
            naive_move=args.iqm_naive_move,
            shots=args.shots,
            circuit=ansatz,
        )
        meta_out.update(iqm_meta)

    energies = []
    exact_energies = []
    details = []

    LOGGER.info(
        "Starting TFIM scan: estimator=%s n_qubits=%s J=%s boundary=%s points=%s",
        args.estimator,
        args.n_qubits,
        args.J,
        args.boundary,
        args.num_points,
    )
    if args.estimator == "aer":
        LOGGER.info("Aer options: shots=%s seed=%s", args.shots, args.seed)
    else:
        LOGGER.info(
            "IQM options: url=%s backend=%s tokens_file=%s naive_move=%s shots=%s",
            meta_out.get("iqm_url"),
            meta_out.get("iqm_backend"),
            args.iqm_tokens_file,
            args.iqm_naive_move,
            args.shots,
        )

    for ratio in ratios:
        h_value = float(ratio * args.J)
        spec = TFIMSpec(n_qubits=args.n_qubits, J=args.J, h=h_value, boundary=args.boundary)
        operator = build_operator(spec)
        LOGGER.info("Running point: h/J=%.6f h=%.6f J=%.6f", ratio, h_value, args.J)
        point_start = time.perf_counter()
        result, meta = run_vqe(operator, ansatz, optimizer, estimator, args.seed)
        point_time = time.perf_counter() - point_start
        payload = result_payload(result, meta)
        energies.append(payload["energy"])
        exact_energy = None
        if args.exact:
            exact_energy = float(np.linalg.eigvalsh(operator.to_matrix())[0].real)
            exact_energies.append(exact_energy)
        LOGGER.info(
            "Completed point: h/J=%.6f energy=%.8f exact=%s time=%.2fs",
            ratio,
            payload["energy"],
            f"{exact_energy:.8f}" if exact_energy is not None else "n/a",
            point_time,
        )
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

    print(json.dumps({"output": args.output, "meta": meta_out, "points": details}, indent=2, sort_keys=True))
    if args.emit_json:
        with open(args.emit_json, "w", encoding="utf-8") as handle:
            json.dump({"output": args.output, "meta": meta_out, "points": details}, handle, indent=2, sort_keys=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
