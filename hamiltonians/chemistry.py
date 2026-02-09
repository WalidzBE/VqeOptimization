from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.analysis import Z2Symmetries
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, FreezeCoreTransformer
from qiskit_nature.units import DistanceUnit

from .base import Hamiltonian, HamiltonianSpec, register_hamiltonian


_H2O_GEOMETRY = "O 0.0 0.0 0.0; H 0.0 0.0 0.958; H 0.9266 0.0 -0.2396"
_SUPPORTED_MOLECULES = {
    "h2o": _H2O_GEOMETRY,
}


@dataclass(frozen=True)
class ChemistrySpec(HamiltonianSpec):
    """Specification for molecular Hamiltonians (currently H2O only)."""

    molecule: str = "h2o"
    basis: str = "sto3g"
    charge: int = 0
    spin: int = 0
    unit: Literal["angstrom"] = "angstrom"
    freeze_core: bool = True
    active_electrons: int = 6
    active_orbitals: int = 5
    mapper: Literal["jw", "parity"] = "parity"
    taper: bool = True
    taper_sector: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", "chemistry")
        object.__setattr__(self, "molecule", self.molecule.lower())

    def validate(self) -> None:
        if self.molecule not in _SUPPORTED_MOLECULES:
            raise ValueError(f"Unsupported molecule: {self.molecule}")
        if self.charge != int(self.charge) or self.spin != int(self.spin):
            raise ValueError("charge and spin must be integers")
        if self.active_electrons <= 0:
            raise ValueError("active_electrons must be > 0")
        if self.active_orbitals <= 0:
            raise ValueError("active_orbitals must be > 0")
        if self.mapper == "jw" and self.taper:
            raise ValueError("tapering requires parity mapper")
        if self.taper_sector < 0:
            raise ValueError("taper_sector must be >= 0")


def _distance_unit(unit: str) -> DistanceUnit:
    if unit == "angstrom":
        return DistanceUnit.ANGSTROM
    raise ValueError(f"Unsupported distance unit: {unit}")


def build_operator(spec: ChemistrySpec) -> SparsePauliOp:
    """Build the molecular qubit Hamiltonian as a SparsePauliOp."""

    spec.validate()
    geometry = _SUPPORTED_MOLECULES[spec.molecule]

    driver = PySCFDriver(
        atom=geometry,
        basis=spec.basis,
        charge=spec.charge,
        spin=spec.spin,
        unit=_distance_unit(spec.unit),
    )

    problem = driver.run()

    if spec.freeze_core:
        problem = FreezeCoreTransformer().transform(problem)

    active_space = ActiveSpaceTransformer(
        num_electrons=spec.active_electrons,
        num_spatial_orbitals=spec.active_orbitals,
    )
    problem = active_space.transform(problem)

    second_q_op = problem.hamiltonian.second_q_op()

    if spec.mapper == "jw":
        mapper = JordanWignerMapper()
    else:
        mapper = ParityMapper()
    qubit_op = mapper.map(second_q_op)

    if spec.mapper == "parity" and spec.taper:
        z2_symmetries = Z2Symmetries.find_z2_symmetries(qubit_op)
        tapered_ops = z2_symmetries.taper(qubit_op)
        if not tapered_ops:
            raise ValueError("No tapered operators produced for the requested symmetry sector")
        if spec.taper_sector >= len(tapered_ops):
            raise ValueError("taper_sector is out of range")
        qubit_op = tapered_ops[spec.taper_sector]

    if isinstance(qubit_op, SparsePauliOp):
        return qubit_op.simplify()
    return SparsePauliOp(qubit_op).simplify()


def default_ansatz(n_qubits: int, reps: int = 2) -> QuantumCircuit:
    """Default hardware-efficient ansatz for molecular VQE runs."""

    return TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=reps,
    )


CHEMISTRY_HAMILTONIAN = Hamiltonian(
    name="chemistry",
    spec_cls=ChemistrySpec,
    build_operator=build_operator,
    default_ansatz=default_ansatz,
)

register_hamiltonian(CHEMISTRY_HAMILTONIAN)
