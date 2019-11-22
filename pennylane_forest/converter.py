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


def _resolve_gate(gate):
    for i, modifier in enumerate(gate.modifiers):
        if modifier == "CONTROLLED":
            if gate.name in _control_map:
                stripped_gate = g.Gate(_control_map[gate.name], gate.params, gate.qubits)
                stripped_gate.modifiers = gate.modifiers.copy()
                del stripped_gate.modifiers[i]

                return _resolve_gate(stripped_gate)
            else:
                break

    return gate


def _get_qubit_index(qubit):
    if isinstance(qubit, int):
        return qubit

    if isinstance(qubit, pyquil.quilatom.Qubit):
        return qubit.index


class ProgramLoader:
    _matrix_dictionary = pyquil.gate_matrices.QUANTUM_GATES

    def _load_defined_gates(self):
        self._defined_gate_names = []

        for defgate in self.program.defined_gates:
            self._defined_gate_names.append(defgate.name)

            if isinstance(defgate, pyquil.quil.DefPermutationGate):
                matrix = np.eye(defgate.permutation.shape[0])
                matrix = matrix[:, defgate.permutation]
            elif isinstance(defgate, pyquil.quil.DefGate):
                matrix = defgate.matrix

            self._matrix_dictionary[defgate.name] = matrix

    def _is_defined_gate(self, gate):
        return gate.name in self._defined_gate_names

    def _is_controlled(self, gate):
        return "CONTROLLED" in gate.modifiers

    def _is_forked(self, gate):
        return "FORKED" in gate.modifiers

    def _is_inverted(self, gate):
        return gate.modifiers.count("DAGGER") % 2 == 1

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
            if self._is_forked(gate):
                raise qml.DeviceError(
                    "Forked gates can not be imported into PennyLane, as this functionality is not supported. "
                    + "Gate Nr. {}, {} was forked.".format(i + 1, gate)
                )

            resolved_gate = _resolve_gate(gate)

            # If the gate is a DefGate or not all CONTROLLED statements can be resolved
            # we resort to QubitUnitary
            if self._is_controlled(resolved_gate) or self._is_defined_gate(resolved_gate):
                pl_gate = lambda wires: qml.QubitUnitary(
                    self._resolve_gate_matrix(resolved_gate), wires=wires
                )
            else:
                pl_gate = pyquil_inv_operation_map[resolved_gate.name]

            wires = self._qubits_to_wires(gate.qubits)
            pl_gate_instance = pl_gate(*gate.params, wires=wires)

            if self._is_inverted(gate):
                pl_gate_instance.inv()


def load_program(program):
    """Load template from PyQuil Program instance."""

    loader = ProgramLoader(program)
    loader.template()
