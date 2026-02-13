"""
simple_knitting.py

Contiene:
- generazione window ansatz (identico a quello usato nel VQE)
- pronto per future operazioni di circuit knitting
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator

    
 
# ============================================================
# WINDOW ANSATZ (IDENTICO A QUELLO USATO PRIMA)
# ============================================================

def build_window_ansatz(n_qubits, window=5, layers=1):

    if window > n_qubits:
        window = n_qubits

    qc = QuantumCircuit(n_qubits)

    # ⭐ initial state |+> (molto importante per TFIM)
    #for q in range(n_qubits):
        #qc.h(q)

    gamma = ParameterVector("γ", layers)
    beta = ParameterVector("β", layers)

    starts = range(0, n_qubits - window + 1)

    for l in range(layers):
        for s in starts:
            qs = list(range(s, s + window))

            # ZZ interactions locali
            for i in range(len(qs) - 1):
                a, b = qs[i], qs[i + 1]
                qc.cx(a, b)
                qc.rz(2 * gamma[l], b)
                qc.cx(a, b)

            # transverse field
            for q in qs:
                qc.rx(2 * beta[l], q)

    return qc










if __name__ == "__main__":

    n = 6
    window = 5
    layers = 1

    ansatz = build_window_ansatz(n, window, layers)

    print("Number of qubits:", ansatz.num_qubits)
    print("Number of parameters:", len(ansatz.parameters))
    print("Depth:", ansatz.depth())
    print(ansatz)
 

    
    
    