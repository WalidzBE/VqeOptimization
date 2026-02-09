from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Callable, Mapping, Sequence, Type, get_args, get_origin, get_type_hints

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


@dataclass(frozen=True)
class HamiltonianSpec:
    """Base specification for Hamiltonian construction."""

    name: str = field(default="", init=False)

    def validate(self) -> None:
        """Validate the spec fields.

        Subclasses should override this method and raise ValueError on invalid inputs.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Hamiltonian:
    name: str
    spec_cls: Type[HamiltonianSpec]
    build_operator: Callable[[HamiltonianSpec], SparsePauliOp]
    default_ansatz: Callable[[int, int], QuantumCircuit] | None = None

    def build_spec(self, params: Mapping[str, Any]) -> HamiltonianSpec:
        spec = self.spec_cls(**params)
        spec.validate()
        return spec


try:
    from typing import Literal, Union  # Python 3.11+ typing
except ImportError:  # pragma: no cover
    from typing_extensions import Literal, Union  # type: ignore


def _literal_choices(annotation: Any) -> Sequence[str] | None:
    origin = get_origin(annotation)
    if origin is Literal:
        return [str(arg) for arg in get_args(annotation)]
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _literal_choices(args[0])
    return None


def _argparse_type(annotation: Any) -> type:
    origin = get_origin(annotation)
    if origin is None:
        if annotation is bool:
            return bool
        return annotation if annotation in (int, float, str) else str
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _argparse_type(args[0])
    return str


def add_hamiltonian_args(parser: Any, ham: Hamiltonian) -> None:
    """Add argparse options for the hamiltonian spec fields."""

    try:
        type_hints = get_type_hints(ham.spec_cls, include_extras=True)
    except TypeError:
        type_hints = get_type_hints(ham.spec_cls)

    for field in fields(ham.spec_cls):
        if field.name == "name":
            continue
        annotation = type_hints.get(field.name, field.type)
        choices = _literal_choices(annotation)
        arg_type = _argparse_type(annotation)
        arg_name = f"--{field.name}"
        kwargs: dict[str, Any] = {}
        if arg_type is bool:
            import argparse

            kwargs["action"] = argparse.BooleanOptionalAction
        elif choices is not None:
            kwargs["choices"] = choices
            kwargs["type"] = str
        else:
            kwargs["type"] = arg_type
        if field.default is not MISSING:
            kwargs["default"] = field.default
        else:
            kwargs["required"] = True
        parser.add_argument(arg_name, **kwargs)


def spec_from_args(ham: Hamiltonian, args: Any) -> HamiltonianSpec:
    params = {}
    for field in fields(ham.spec_cls):
        if field.name == "name":
            continue
        params[field.name] = getattr(args, field.name)
    return ham.build_spec(params)


_HAMILTONIANS: dict[str, Hamiltonian] = {}


def register_hamiltonian(ham: Hamiltonian) -> None:
    if ham.name in _HAMILTONIANS:
        raise ValueError(f"Hamiltonian '{ham.name}' already registered")
    _HAMILTONIANS[ham.name] = ham


def get_hamiltonian(name: str) -> Hamiltonian:
    if name not in _HAMILTONIANS:
        raise KeyError(f"Unknown hamiltonian: {name}")
    return _HAMILTONIANS[name]


def list_hamiltonians() -> Sequence[str]:
    return sorted(_HAMILTONIANS.keys())
