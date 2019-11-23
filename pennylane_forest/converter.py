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

    def _load_defined_gate_names(self):
        self._defined_gate_names = []

        for defgate in self.program.defined_gates:
            self._defined_gate_names.append(defgate.name)

            if isinstance(defgate, pyquil.quil.DefPermutationGate):
                matrix = np.eye(defgate.permutation.shape[0])
                matrix = matrix[:, defgate.permutation]
            elif isinstance(defgate, pyquil.quil.DefGate):
                matrix = defgate.matrix

            self._matrix_dictionary[defgate.name] = matrix

    def _load_declarations(self):
        self._declarations = [
            instruction
            for instruction in self.program.instructions
            if self._is_declaration(instruction)
        ]

    def _is_defined_gate(self, gate):
        return gate.name in self._defined_gate_names

    def _is_controlled(self, gate):
        return "CONTROLLED" in gate.modifiers

    def _is_forked(self, gate):
        return "FORKED" in gate.modifiers

    def _is_inverted(self, gate):
        return gate.modifiers.count("DAGGER") % 2 == 1

    def _is_declaration(self, instruction):
        return isinstance(instruction, pyquil.quil.Declare)

    def _load_qubit_to_wire_map(self, wires):
        if len(wires) != len(self.qubits):
            raise qml.DeviceError(
                "The number of given wires does not match the number of qubits in the PyQuil Program. "
                + "{} wires were given, Program has {} qubits".format(len(wires), len(self.qubits))
            )

        self._qubit_to_wire_map = dict(zip(self.qubits, wires))

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

    def _normalize_variable_map(self, variable_map):
        new_keys = list(variable_map.keys())
        values = list(variable_map.values())

        for i in range(len(new_keys)):
            if isinstance(new_keys[i], pyquil.quil.MemoryReference):
                new_keys[i] = new_keys[i].name

        return dict(zip(new_keys, values))

    def _check_variable_map(self, variable_map):
        for declaration in self._declarations:
            if not declaration.name in variable_map:
                raise qml.DeviceError(
                    "The PyQuil program defines a variable {} that is not present in the given variable map. "
                    + "Instruction: {}".format(declaration.name, declaration)
                )

    def _resolve_params(self, params, variable_map):
        resolved_params = []

        for param in params:
            if isinstance(param, pyquil.quilatom.MemoryReference):
                resolved_params.append(variable_map[param.name])
            else:
                resolved_params.append(param)

        return resolved_params

    def __init__(self, program):
        self.program = program
        self.qubits = program.get_qubits()

        self._load_defined_gate_names()
        self._load_declarations()

    @property
    def defined_gates(self):
        return self.program.defined_gates

    @property
    def declarations(self):
        return self._declarations

    def template(self, variable_map={}, wires=None):
        if not wires:
            wires = range(len(self.qubits))

        variable_map = self._normalize_variable_map(variable_map)

        self._load_qubit_to_wire_map(wires)

        self._check_variable_map(variable_map)

        for i, instruction in enumerate(self.program.instructions):
            if self._is_declaration(instruction):
                continue

            # Rename for better readability
            gate = instruction

            if self._is_forked(gate):
                raise qml.DeviceError(
                    "Forked gates can not be imported into PennyLane, as this functionality is not supported. "
                    + "Instruction Nr. {}, {} was a forked gate.".format(i + 1, gate)
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

            target_wires = self._qubits_to_wires(gate.qubits)
            resolved_params = self._resolve_params(gate.params, variable_map)

            pl_gate_instance = pl_gate(*resolved_params, wires=target_wires)

            if self._is_inverted(gate):
                pl_gate_instance.inv()

    def __call__(self, variable_map={}, wires=None):
        self.template(variable_map, wires)


def load_program(program: pyquil.Program):
    """Load template from PyQuil Program instance."""

    return ProgramLoader(program)


def load_quil(quil_str: str):

    return load_program(pyquil.Program(quil_str))


def load_quil_from_file(file_path: str):
    with open(file_path, 'r') as file:
        quil_str = file.read()

    return load_quil(quil_str)