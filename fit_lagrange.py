from qiskit import QuantumCircuit
import numpy as np




# ============================================
# HARDWARE LIMIT (LAGRANGE)
# ============================================

MAX_QUBITS = 5

# ============================================
# FIDELITY
# ============================================

from qiskit_aer import AerSimulator


def measurement_distribution(circuit, shots=10000):
    sim = AerSimulator()

    circ = circuit.copy()
    
    if not any(op.operation.name == "measure" for op in circ.data):
        circ.measure_all()
   

    job = sim.run(circ, shots=shots)
    result = job.result()

    return result.get_counts()


def compare_distributions(c1, c2):
    print("\nOriginal:", measurement_distribution(c1))
    print("Deferred:", measurement_distribution(c2))




from qiskit_aer import AerSimulator

def has_measurements(qc):
    return any(inst.operation.name == "measure" for inst in qc.data)

def get_counts_safely(qc, shots=20000, seed=1234):
    sim = AerSimulator(seed_simulator=seed)
    circ = qc.copy()

    #  se ha parametri → assegna valori casuali
    if circ.parameters:
        rng = np.random.default_rng(seed)
        bind = {p: rng.uniform(0, 2*np.pi) for p in circ.parameters}
        circ = circ.assign_parameters(bind)

    if not has_measurements(circ):
        circ.measure_all()

    result = sim.run(circ, shots=shots).result()
    return result.get_counts()

def tv_distance(counts_a, counts_b):
    """Total variation distance tra due distribuzioni discrete."""
    shots_a = sum(counts_a.values()) or 1
    shots_b = sum(counts_b.values()) or 1

    keys = set(counts_a) | set(counts_b)
    dist = 0.0
    for k in keys:
        pa = counts_a.get(k, 0) / shots_a
        pb = counts_b.get(k, 0) / shots_b
        dist += abs(pa - pb)
    return 0.5 * dist

def validate_equivalence(original, candidate, shots=20000, tv_tol=0.02, seed=1234, verbose=True):
    """
    Confronta le distribuzioni di misura. Se tv_distance <= tv_tol -> ok.
    tv_tol ~ 0.01-0.03 è ragionevole con 20k shot.
    """
    c0 = get_counts_safely(original, shots=shots, seed=seed)
    c1 = get_counts_safely(candidate, shots=shots, seed=seed)

    d = tv_distance(c0, c1)
    if verbose:
        print("\n--- VALIDATION ---")
        print("TV distance:", d)
        print("Original:", c0)
        print("Candidate:", c1)

    return d <= tv_tol












# ============================================
# CHECK HARDWARE CONSTRAINT
# ============================================

def check_qubits(circuit):
    print("\n--- CHECKING HARDWARE CONSTRAINT ---")

    n = circuit.num_qubits
    print("Number of qubits in circuit:", n)
    print("Lagrange max qubits:", MAX_QUBITS)

    if n <= MAX_QUBITS:
        print("Circuit FITS Lagrange hardware")
        return True
    else:
        print("Circuit TOO LARGE for Lagrange")
        return False



def qubits_used_later(circuit, start_index):
    """
    Returns set of qubit indices used after instruction index.
    """
    used = set()

    for inst in circuit.data[start_index + 1:]:
        for q in inst.qubits:
            used.add(circuit.find_bit(q).index)

    return used


# ============================================
# TRY DEFERRED MEASUREMENT (placeholder)
# ============================================

from qiskit import QuantumCircuit

ENTANGLING_NAMES = {
    # non serve essere perfetti: tutto ciò che è multi-qubit lo trattiamo come “rischioso”
    # puoi lasciare vuoto e usare solo len(qargs)>1
}
from qiskit import QuantumCircuit

def compute_last_use(circuit):
    n = circuit.num_qubits
    last = [-1] * n
    for i, inst in enumerate(circuit.data):
        for q in inst.qubits:
            l = circuit.find_bit(q).index
            last[l] = i
    return last

def is_multi_qubit_gate(op, qargs):
    if op.name in ("barrier", "measure", "reset"):
        return False
    return len(qargs) > 1

def deferred_reuse(circuit, max_physical=None, mode="heuristic", verbose=True):
    """
    Restituisce un circuito equivalente (heuristic) o conservativo (safe)
    usando measure+reset+reuse.
    Output: circuito con peak_physical qubit e n_logical bit classici (logici).
    """
    if verbose:
        print(f"\n--- APPLYING DEFERRED MEASUREMENT ({mode.upper()}) + RESET + QUBIT REUSE ---")

    n_logical = circuit.num_qubits
    last_use = compute_last_use(circuit)

    # neighbors graph (solo per safe)
    neighbors = [set() for _ in range(n_logical)]
    freed = set()

    logical_to_physical = {}
    free_physical = []
    next_physical = 0
    peak_physical = 0

    scheduled = []  # (op, phys_qargs) OR ("measure", phys, logical) OR ("reset", phys)

    def get_phys(logical):
        nonlocal next_physical
        if logical in logical_to_physical:
            return logical_to_physical[logical]
        if free_physical:
            phys = free_physical.pop()
            if verbose:
                print(f"Reusing physical {phys} for logical {logical}")
        else:
            phys = next_physical
            next_physical += 1
        logical_to_physical[logical] = phys
        return phys

    def live_set(after_index):
        return {q for q in range(n_logical) if last_use[q] > after_index and q not in freed}

    for i, inst in enumerate(circuit.data):
        op = inst.operation
        qargs = inst.qubits
        logicals = [circuit.find_bit(q).index for q in qargs]
        phys_qargs = [get_phys(l) for l in logicals]

        scheduled.append((op, phys_qargs))
        peak_physical = max(peak_physical, next_physical)

        # aggiorna neighbors se gate multi-qubit (serve solo al safe)
        if is_multi_qubit_gate(op, qargs):
            for a in logicals:
                for b in logicals:
                    if a != b:
                        neighbors[a].add(b)

        # prova a liberare logical che finiscono qui
        live = live_set(i)
        for l in set(logicals):
            if last_use[l] != i:
                continue

            if mode == "safe":
                # non liberare se ha neighbor ancora vivi
                if neighbors[l].intersection(live):
                    if verbose:
                        print(f"Logical {l} finished but has live neighbors {neighbors[l].intersection(live)} → NOT freeing")
                    continue

            # heuristic: libera sempre se last_use==i
            phys = logical_to_physical[l]
            if verbose:
                print(f"Logical {l} finished → measure/reset physical {phys}")

            scheduled.append(("measure", phys, l))  # misura su bit classico LOGICAL
            scheduled.append(("reset", phys))

            free_physical.append(phys)
            del logical_to_physical[l]
            freed.add(l)

    # misura finale per i logical rimasti vivi
    for l, phys in list(logical_to_physical.items()):
        if verbose:
            print(f"Logical {l} still alive at end → final measure physical {phys}")
        scheduled.append(("measure", phys, l))
        del logical_to_physical[l]

    if verbose:
        print("Peak physical qubits used:", peak_physical)

    if max_physical and peak_physical > max_physical:
        raise RuntimeError(f"Circuit requires {peak_physical} qubits but limit is {max_physical}")

    new_circ = QuantumCircuit(peak_physical, n_logical)
    for item in scheduled:
        if item[0] == "measure":
            _, phys, logical = item
            new_circ.measure(phys, logical)
        elif item[0] == "reset":
            _, phys = item
            new_circ.reset(phys)
        else:
            op, phys_qargs = item
            new_circ.append(op, phys_qargs)

    return new_circ


def deferred_with_rollback(circuit, max_physical, shots=20000, tv_tol=0.02, seed=1234, verbose=True):
    """
    1) prova heuristic (più riduzione)
       - se entra in max_physical e valida -> ok
    2) altrimenti prova safe
       - se valida -> ok
    3) altrimenti fallback: circuito originale (o errore)
    """
    # Try heuristic
    try:
        cand = deferred_reuse(circuit, max_physical=max_physical, mode="heuristic", verbose=verbose)
        ok = validate_equivalence(circuit, cand, shots=shots, tv_tol=tv_tol, seed=seed, verbose=verbose)
        if ok:
            if verbose:
                print("✅ Heuristic deferred PASSED validation")
            return cand
        else:
            if verbose:
                print("❌ Heuristic deferred FAILED validation → trying SAFE")
    except RuntimeError as e:
        if verbose:
            print("Heuristic deferred cannot fit:", e)
        cand = None

    # Try safe
    try:
        cand2 = deferred_reuse(circuit, max_physical=max_physical, mode="safe", verbose=verbose)
        ok2 = validate_equivalence(circuit, cand2, shots=shots, tv_tol=tv_tol, seed=seed, verbose=verbose)
        if ok2:
            if verbose:
                print("✅ SAFE deferred PASSED validation")
            return cand2
        else:
            if verbose:
                print("❌ SAFE deferred FAILED validation → fallback")
    except RuntimeError as e:
        if verbose:
            print("SAFE deferred cannot fit:", e)

    # fallback
    return None  # oppure: return circuit



# ============================================
# TRY CIRCUIT CUTTING (placeholder)
# ============================================

def try_circuit_cutting(circuit):
    print("\n--- TRYING CIRCUIT CUTTING ---")
    print("Not implemented yet")
    return None


# ============================================
# MAIN FUNCTION
# ============================================

def fit_for_lagrange(circuit):
    print("\n====================================")
    print("STARTING LAGRANGE FITTING")
    print("====================================")

    # STEP 1
    fits = check_qubits(circuit)

    if fits:
        print("\nNothing to do.")
        return circuit

    # STEP 2
    #new_circuit = try_deferred_measurement(circuit)
    # if new_circuit:
        # print("Reduced with deferred measurement")
        # return new_circuit

    # STEP 3
    new_circuit = try_circuit_cutting(circuit)
    if new_circuit:
        print("Reduced with circuit cutting")
        return new_circuit

    print("\n Cannot fit circuit to 5 qubits.")
    return None






    
   

def fit_for_lagrange(circuit):
    print("\n====================================")
    print("STARTING LAGRANGE FITTING")
    print("====================================")

    # if circuit.num_qubits <= MAX_QUBITS:
        # print("Circuit fits Lagrange hardware. Nothing to do.")
        # return circuit

    # STEP 1: deferred + rollback
    reduced = deferred_with_rollback(
        circuit,
        max_physical=MAX_QUBITS,
        shots=20000,
        tv_tol=0.02,
        seed=1234,
        verbose=True
    )
    
    print(circuit.decompose())
    
    if reduced is None:
        print("\nCannot fit circuit to 5 qubits (deferred failed; cutting not implemented).")
        return None

    print("\nReduced circuit:")
    print(reduced.decompose())
    
    return reduced
    
  
    # STEP 2: circuit cutting (placeholder)
    # reduced2 = try_circuit_cutting(circuit)
    # if reduced2 is not None: return reduced2

 
# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    

    print("Creating test circuit with 7 qubits...")

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0,2)
    qc.cx(1,2)
    
    fit_for_lagrange(qc)

    #qc_def = try_deferred_measurement(qc)

    #print(qc)
    #print(qc_def)


    #print("\n=== Compare distributions ===")
    #compare_distributions(qc, qc_def)
    