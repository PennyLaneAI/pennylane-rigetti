from collections.abc import Sequence
import numpy as np

import pennylane as qml
import pennylane_forest as plf
import pyquil
import pyquil.gates as g

pyquil_inv_operation_map = {
    "X": qml.PauliX,
    "Y": qml.PauliY,
    "Z": qml.PauliZ,
    "H": qml.Hadamard,
    "CNOT": qml.CNOT,
    "SWAP": qml.SWAP,
    "CZ": qml.CZ,
    "PHASE": qml.PhaseShift,
    "RX": qml.RX,
    "RY": qml.RY,
    "RZ": qml.RZ,
    "CRX": qml.CRX,
    "CRY": qml.CRY,
    "CRZ": qml.CRZ,
    "S": plf.ops.S,
    "T": plf.ops.T,
    "CCNOT": plf.ops.CCNOT,
    "CPHASE": lambda *params, wires: plf.ops.CPHASE(*params, 3, wires=wires),
    "CPHASE00": lambda *params, wires: plf.ops.CPHASE(*params, 0, wires=wires),
    "CPHASE01": lambda *params, wires: plf.ops.CPHASE(*params, 1, wires=wires),
    "CPHASE10": lambda *params, wires: plf.ops.CPHASE(*params, 2, wires=wires),
    "CSWAP": plf.ops.CSWAP,
    "ISWAP": plf.ops.ISWAP,
    "PSWAP": plf.ops.PSWAP,
}

_control_map = {
    "X": "CNOT",
    "Z": "CZ",
    "CNOT": "CCNOT",
    "SWAP": "CSWAP",
    "RX": "CRX",
    "RY": "CRY",
    "RZ": "CRZ",
    "PHASE": "CPHASE",
}

_matrix_dictionary = pyquil.gate_matrices.QUANTUM_GATES

def _direct_sum(A, B):
    sum = np.zeros(np.add(A.shape, B.shape), dtype=A.dtype)
    sum[: A.shape[0], : A.shape[1]] = A
    sum[A.shape[0] :, A.shape[1] :] = B

    return sum

def _controlled_matrix(op):
    return _direct_sum(np.eye(op.shape[0], dtype=op.dtype), op)

def _simplify_controlled_operations(gate):
    print("_simplify_controlled_operations/gate = ", gate)
    for i, modifier in enumerate(gate.modifiers):
        if modifier == "CONTROLLED":
            print("_simplify_controlled_operations/i, modifier = ", i, ", ", modifier)
            if gate.name in _control_map:
                stripped = g.Gate(_control_map[gate.name], gate.params, gate.qubits)
                stripped.modifiers = gate.modifiers.copy()
                del stripped.modifiers[i]

                print("stripped.modifiers = ", stripped.modifiers)

                return _simplify_controlled_operations(stripped)
            else:
                break

    return gate


def _get_qubit_index(qubit):
    if isinstance(qubit, int):
        return qubit

    if isinstance(qubit, pyquil.quilatom.Qubit):
        return qubit.index

    raise Exception("I can't get that Qubit index.")




class ProgramLoader:
    _matrix_dictionary = pyquil.gate_matrices.QUANTUM_GATES

    def _load_defined_gates(self):
        self.defgate_to_matrix_map = {}

        for defgate in self.program.defined_gates:
            if isinstance(defgate, pyquil.quil.DefPermutationGate):
                permutation_matrix = np.eye(defgate.permutation.shape[0])
                permutation_matrix = permutation_matrix[:, defgate.permutation]
                self.defgate_to_matrix_map[defgate.name] = permutation_matrix
            elif isinstance(defgate, pyquil.quil.DefGate):
                self.defgate_to_matrix_map[defgate.name] = defgate.matrix

            self._matrix_dictionary[defgate.name] = self.defgate_to_matrix_map[defgate.name]
    
    def _qubits_to_wires(self, qubits):
        if isinstance(qubits, Sequence):
            return [self._qubit_to_wire_map[_get_qubit_index(qubit)] for qubit in qubits]

        return self._qubit_to_wire_map[_get_qubit_index(qubits)]

    def _resolve_gate_matrix(self, gate):
        gate_matrix = self._matrix_dictionary[gate.name]
        
        for i, modifier in enumerate(gate.modifiers):
            if modifier == "CONTROLLED":
                gate_matrix = _controlled_matrix(gate_matrix)

        return gate_matrix

    def __init__(self, program):
        self.program = program

        self._qubit_to_wire_map = dict(zip(program.get_qubits(), range(len(program.get_qubits()))))
        self._load_defined_gates()

    def template(self):
        for i, gate in enumerate(self.program.instructions):
            if "FORKED" in gate.modifiers:
                raise qml.DeviceError(
                    "Forked gates can not be imported into PennyLane, as this functionality is not supported. "
                    + "Gate Nr. {}, {} was forked.".format(i + 1, gate)
                )

            simplified_gate = _simplify_controlled_operations(gate)

            if "CONTROLLED" in simplified_gate.modifiers or simplified_gate.name in self.defgate_to_matrix_map:
                pl_gate = lambda wires: qml.QubitUnitary(
                    self._resolve_gate_matrix(simplified_gate), wires=wires
                )
            else:
                pl_gate = pyquil_inv_operation_map[simplified_gate.name]

            wires = self._qubits_to_wires(gate.qubits)
            pl_gate_instance = pl_gate(*gate.params, wires=wires)

            if gate.modifiers.count("DAGGER") % 2 == 1:
                pl_gate_instance.inv()
    


def load_program(program):
    """Load template from PyQuil Program instance."""

    loader = ProgramLoader(program)
    loader.template()