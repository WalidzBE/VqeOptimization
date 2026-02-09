import pytest

from hamiltonians.tfim import TFIMSpec, build_operator


def _terms(op):
    return {pauli: float(coeff.real) for pauli, coeff in op.to_list()}


def test_tfim_open_n2_terms():
    spec = TFIMSpec(n_qubits=2, J=1.5, h=0.7, boundary="open")
    op = build_operator(spec)
    terms = _terms(op)
    assert terms["ZZ"] == pytest.approx(-1.5)
    assert terms["IX"] == pytest.approx(-0.7)
    assert terms["XI"] == pytest.approx(-0.7)
    assert len(terms) == 3


def test_tfim_open_n3_terms():
    spec = TFIMSpec(n_qubits=3, J=2.0, h=0.5, boundary="open")
    op = build_operator(spec)
    terms = _terms(op)

    assert terms["IZZ"] == pytest.approx(-2.0)
    assert terms["ZZI"] == pytest.approx(-2.0)

    assert terms["IIX"] == pytest.approx(-0.5)
    assert terms["IXI"] == pytest.approx(-0.5)
    assert terms["XII"] == pytest.approx(-0.5)
    assert len(terms) == 5


def test_tfim_periodic_n3_terms():
    spec = TFIMSpec(n_qubits=3, J=1.0, h=0.0, boundary="periodic")
    op = build_operator(spec)
    terms = _terms(op)

    assert terms["IZZ"] == pytest.approx(-1.0)
    assert terms["ZZI"] == pytest.approx(-1.0)
    assert terms["ZIZ"] == pytest.approx(-1.0)
    assert len(terms) == 3


def test_tfim_invalid_n_qubits():
    with pytest.raises(ValueError):
        build_operator(TFIMSpec(n_qubits=1, J=1.0, h=0.5, boundary="open"))


def test_tfim_invalid_boundary():
    with pytest.raises(ValueError):
        build_operator(TFIMSpec(n_qubits=2, J=1.0, h=0.5, boundary="ring"))
