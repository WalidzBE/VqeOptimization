from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Callable

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals

import hamiltonians  # noqa: F401
from hamiltonians.base import add_hamiltonian_args, get_hamiltonian, spec_from_args


LOGGER = logging.getLogger(__name__)
DEFAULT_IQM_URL = "https://spark.quantum.linksfoundation.com/station"

ANSATZ_FACTORIES: dict[str, Callable[[int, int], QuantumCircuit]] = {
    "efficient_su2": lambda n, reps: EfficientSU2(num_qubits=n, reps=reps, entanglement="linear"),
    "two_local": lambda n, reps: TwoLocal(
        num_qubits=n,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cx",
        reps=reps,
        entanglement="linear",
    ),
}

OPTIMIZER_FACTORIES: dict[str, Callable[[int, int | None], Any]] = {
    "cobyla": lambda maxiter, seed: COBYLA(maxiter=maxiter),
    "spsa": lambda maxiter, seed: SPSA(maxiter=maxiter, seed=seed),
}


def build_base_parser(description: str, add_hamiltonian: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, add_help=False, allow_abbrev=False)
    if add_hamiltonian:
        parser.add_argument("--hamiltonian", default="tfim")
    parser.add_argument("--ansatz", default="efficient_su2", choices=sorted(ANSATZ_FACTORIES.keys()))
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--optimizer", default="cobyla", choices=sorted(OPTIMIZER_FACTORIES.keys()))
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def add_iqm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--estimator", default="aer", choices=["aer", "iqm"])
    parser.add_argument("--iqm_url", default=None)
    parser.add_argument("--iqm_backend", default=None)
    parser.add_argument("--iqm_tokens_file", default=None)
    parser.add_argument("--iqm_naive_move", action=argparse.BooleanOptionalAction, default=False)


def build_parser_with_hamiltonian(description: str, argv: list[str]) -> argparse.Namespace:
    base = build_base_parser(description)
    pre_args, _ = base.parse_known_args(argv)
    ham = get_hamiltonian(pre_args.hamiltonian)

    parser = argparse.ArgumentParser(description=description, parents=[base], allow_abbrev=False)
    add_hamiltonian_args(parser, ham)
    return parser.parse_args(argv)


def build_spec_and_operator(args: argparse.Namespace) -> tuple[SparsePauliOp, int]:
    ham = get_hamiltonian(args.hamiltonian)
    spec = spec_from_args(ham, args)
    operator = ham.build_operator(spec)
    n_qubits = getattr(spec, "n_qubits", None)
    if n_qubits is None:
        n_qubits = operator.num_qubits
    return operator, n_qubits


def build_ansatz(name: str, n_qubits: int, reps: int) -> QuantumCircuit:
    if name not in ANSATZ_FACTORIES:
        raise ValueError(f"Unknown ansatz: {name}")
    return ANSATZ_FACTORIES[name](n_qubits, reps)


def build_optimizer(name: str, maxiter: int, seed: int | None) -> Any:
    if name not in OPTIMIZER_FACTORIES:
        raise ValueError(f"Unknown optimizer: {name}")
    return OPTIMIZER_FACTORIES[name](maxiter, seed)


def build_iqm_estimator(
    *,
    iqm_url: str | None,
    iqm_backend: str | None,
    iqm_tokens_file: str | None,
    naive_move: bool,
    shots: int | None,
    circuit: QuantumCircuit | None = None,
) -> tuple[Any, QuantumCircuit | None, dict[str, Any]]:
    try:
        from iqm.qiskit_iqm import IQMProvider, iqm_naive_move_pass
        from qiskit.primitives import BackendEstimator
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "iqm-client[qiskit] is required for IQM hardware runs. "
            "Install it with: uv pip install iqm-client[qiskit]==29.14"
        ) from exc

    if iqm_tokens_file:
        os.environ["IQM_TOKENS_FILE"] = iqm_tokens_file

    resolved_url = iqm_url or os.environ.get("IQM_URL") or DEFAULT_IQM_URL
    provider = IQMProvider(resolved_url)
    backend = provider.get_backend(iqm_backend) if iqm_backend else provider.get_backend()

    options = {"shots": shots} if shots is not None else None
    estimator = BackendEstimator(
        backend=backend,
        options=options,
        skip_transpilation=naive_move,
    )

    updated_circuit = circuit
    if circuit is not None and naive_move:
        updated_circuit = iqm_naive_move_pass.transpile_to_IQM(circuit, backend=backend)

    backend_name = getattr(backend, "name", None)
    backend_label = backend_name() if callable(backend_name) else backend_name or str(backend)

    meta = {
        "iqm_url": resolved_url,
        "iqm_backend": backend_label,
        "iqm_naive_move": naive_move,
    }
    return estimator, updated_circuit, meta


def run_vqe(
    operator: SparsePauliOp,
    ansatz: QuantumCircuit,
    optimizer: Any,
    estimator: Any,
    seed: int | None,
) -> tuple[Any, dict[str, Any]]:
    if seed is not None:
        algorithm_globals.random_seed = seed
        np.random.seed(seed)
        initial_point = np.random.random(ansatz.num_parameters)
    else:
        initial_point = None

    eval_times: list[float] = []
    start = time.perf_counter()
    last_eval = start

    def callback(eval_count: int, params: np.ndarray, mean: float, meta_or_stddev: Any) -> None:
        nonlocal last_eval
        now = time.perf_counter()
        eval_times.append(now - last_eval)
        last_eval = now
        if LOGGER.isEnabledFor(logging.INFO):
            LOGGER.info("VQE eval %s", eval_count)

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
        callback=callback,
    )
    result = vqe.compute_minimum_eigenvalue(operator)
    total_time = time.perf_counter() - start

    meta = {
        "total_time_s": total_time,
        "eval_time_s": eval_times,
        "optimizer_evals": getattr(result, "optimizer_evals", None),
    }
    return result, meta


def result_payload(result: Any, meta: dict[str, Any]) -> dict[str, Any]:
    optimal_params = getattr(result, "optimal_parameters", None)
    if optimal_params is not None:
        optimal_params = {str(k): float(v) for k, v in optimal_params.items()}
    payload = {
        "energy": float(getattr(result, "optimal_value", result.eigenvalue)),
        "optimal_parameters": optimal_params,
        "optimizer_evals": getattr(result, "optimizer_evals", None),
        "metadata": meta,
    }
    return payload


def transpile_metrics(circuit: QuantumCircuit, seed: int | None) -> dict[str, Any]:
    start = time.perf_counter()
    transpiled = transpile(circuit, optimization_level=1, seed_transpiler=seed)
    transpile_time = time.perf_counter() - start

    one_qubit = 0
    two_qubit = 0
    for inst, qargs, _ in transpiled.data:
        n_qubits = inst.num_qubits
        if n_qubits == 1:
            one_qubit += 1
        elif n_qubits == 2:
            two_qubit += 1

    return {
        "transpile_time_s": transpile_time,
        "depth": transpiled.depth(),
        "one_qubit_gates": one_qubit,
        "two_qubit_gates": two_qubit,
        "count_ops": {k: int(v) for k, v in transpiled.count_ops().items()},
    }
