from fit_lagrange import fit_for_lagrange
from qiskit.circuit.library import EfficientSU2

# importa il tuo ansatz
from tfim_trotter import tfim_window_ansatz


n = 6

# print("\n==============================")
# print("EFFICIENT SU2")
# print("==============================")

# ansatz1 = EfficientSU2(n, reps=2, entanglement="linear")
# r1 = fit_for_lagrange(ansatz1)

# print("Reducible:", r1 is not None)


print("\n==============================")
print("TFIM TROTTER ANSATZ")
print("==============================")

ansatz = tfim_window_ansatz(6, window=5, layers=1)
reduced = fit_for_lagrange(ansatz)
print("Reducible:", reduced is not None)
