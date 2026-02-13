from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def tfim_window_ansatz(n_qubits, window=5, layers=2):
    """
    Ansatz a finestra (sliding window) per favorire qubit reuse e cutting.
    Applica interazioni ZZ + field X solo su sottogruppi contigui di dimensione 'window'.
    """
    if window > n_qubits:
        window = n_qubits

    qc = QuantumCircuit(n_qubits)

    gamma = ParameterVector("γw", layers)
    beta  = ParameterVector("βw", layers)

    # finestre contigue
    starts = list(range(0, n_qubits - window + 1))

    for l in range(layers):
        for s in starts:
            qs = list(range(s, s + window))

            # ZZ chain dentro la finestra
            for i in range(len(qs) - 1):
                a, b = qs[i], qs[i+1]
                qc.cx(a, b)
                qc.rz(2 * gamma[l], b)
                qc.cx(a, b)

            # X field dentro la finestra
            for q in qs:
                qc.rx(2 * beta[l], q)

    return qc
