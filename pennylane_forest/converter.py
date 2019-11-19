from collections.abc import Sequence
import pennylane as qml
import pennylane_forest as plf
import pyquil
import pyquil.gates as g

pyquil_inv_operation_map = {
    "X" : qml.PauliX,
    "Y" : qml.PauliY,
    "Z" : qml.PauliZ,
    "H" : qml.Hadamard,
    "CNOT" : qml.CNOT,
    "SWAP" : qml.SWAP,
    "CZ" : qml.CZ,
    "PHASE" : qml.PhaseShift,
    "RX" : qml.RX,
    "RY" : qml.RY,
    "RZ" : qml.RZ,

    # the following gates are provided by the PL-Forest plugin
    "S" : plf.ops.S,
    "T" : plf.ops.T,
    "CCNOT" : plf.ops.CCNOT,
    "CPHASE" : plf.ops.CPHASE,
    "CSWAP" : plf.ops.CSWAP,
    "ISWAP" : plf.ops.ISWAP,
    "PSWAP" : plf.ops.PSWAP,
}

def _get_qubit_index(qubit):
    if isinstance(qubit, int):
        return qubit

    if isinstance(qubit, pyquil.quilatom.Qubit):
        return qubit.index

    raise Exception("I can't get that Qubit index.")

def load_program(program):
    """Load template from PyQuil Program instance."""

    program_qubits = program.get_qubits()
    print("program.qet_qubits() = ", program_qubits)
    qubit_to_wire_map = dict(zip(program_qubits, range(len(program_qubits))))
    print("qubit_to_wire_map = ", qubit_to_wire_map)

    def _qubits_to_wires(qubits):
        if isinstance(qubits, Sequence):
            return [qubit_to_wire_map[_get_qubit_index(qubit)] for qubit in qubits]
        
        return qubit_to_wire_map[_get_qubit_index(qubits)]

    for i, gate in enumerate(program.instructions):
        print("gate[{}] = {}".format(i, gate))
        # print("gate.modifiers = ", gate.modifiers)
        # print("gate.name = ", gate.name)
        # print("gate.qubits = ", gate.qubits)
        # print(" -> wires = ", _qubits_to_wires(gate.qubits))
        # print("gate.params = ", gate.params)

        pl_gate = pyquil_inv_operation_map[gate.name]
        
        wires = _qubits_to_wires(gate.qubits)
        pl_gate_instance = pl_gate(*gate.params, wires=wires)

        if "DAGGER" in gate.modifiers:
            pl_gate_instance.inv()
