from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

from .base import Hamiltonian, HamiltonianSpec, register_hamiltonian


@dataclass(frozen=True)
class TFIMSpec(HamiltonianSpec):
    """Specification for the Transverse-Field Ising Model Hamiltonian."""

    n_qubits: int
    J: float
    h: float
    boundary: Literal["open", "periodic"] = "open"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", "tfim")

    def validate(self) -> None:
        if self.n_qubits < 2:
            raise ValueError("n_qubits must be >= 2")
        if self.boundary not in ("open", "periodic"):
            raise ValueError("boundary must be 'open' or 'periodic'")
        if not math.isfinite(self.J) or not math.isfinite(self.h):
            raise ValueError("J and h must be finite")


def _pauli_string(n_qubits: int, x_indices: Iterable[int] = (), z_indices: Iterable[int] = ()) -> str:
    ops = ["I"] * n_qubits
    for index in x_indices:
        ops[n_qubits - 1 - index] = "X"
    for index in z_indices:
        ops[n_qubits - 1 - index] = "Z"
    return "".join(ops)


def build_operator(spec: TFIMSpec) -> SparsePauliOp:
    """Build the TFIM Hamiltonian operator as a SparsePauliOp."""

    spec.validate()
    terms: list[tuple[str, float]] = []

    if spec.J != 0.0:
        max_edge = spec.n_qubits - 1 if spec.boundary == "open" else spec.n_qubits
        for i in range(max_edge):
            j = (i + 1) % spec.n_qubits
            pauli = _pauli_string(spec.n_qubits, z_indices=(i, j))
            terms.append((pauli, -spec.J))

    if spec.h != 0.0:
        for i in range(spec.n_qubits):
            pauli = _pauli_string(spec.n_qubits, x_indices=(i,))
            terms.append((pauli, -spec.h))

    if not terms:
        return SparsePauliOp.from_list([("I" * spec.n_qubits, 0.0)])
    return SparsePauliOp.from_list(terms).simplify()


def default_ansatz(n_qubits: int, reps: int = 2) -> QuantumCircuit:
    """Default ansatz for TFIM VQE runs."""

    return EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement="linear")


TFIM_HAMILTONIAN = Hamiltonian(
    name="tfim",
    spec_cls=TFIMSpec,
    build_operator=build_operator,
    default_ansatz=default_ansatz,
)

register_hamiltonian(TFIM_HAMILTONIAN)
