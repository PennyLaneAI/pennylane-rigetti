from collections.abc import Sequence
import numpy as np

import pennylane as qml
import pennylane_forest as plf
import pyquil
import pyquil.gates as g

import warnings

pyquil_inv_operation_map = {
    "I": qml.Identity,
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


def _resolve_params(params, variable_map):
    resolved_params = []

    for param in params:
        if isinstance(param, pyquil.quilatom.MemoryReference):
            resolved_params.append(variable_map[param.name])
        else:
            resolved_params.append(param)

    return resolved_params


def _normalize_variable_map(variable_map):
    new_keys = list(variable_map.keys())
    values = list(variable_map.values())

    for i in range(len(new_keys)):
        if isinstance(new_keys[i], pyquil.quil.MemoryReference):
            new_keys[i] = new_keys[i].name

    return dict(zip(new_keys, values))


def _is_controlled(gate):
    return "CONTROLLED" in gate.modifiers


def _is_forked(gate):
    return "FORKED" in gate.modifiers


def _is_inverted(gate):
    return gate.modifiers.count("DAGGER") % 2 == 1


def _is_gate(instruction):
    return isinstance(instruction, pyquil.quil.Gate)


def _is_declaration(instruction):
    return isinstance(instruction, pyquil.quil.Declare)


def _is_measurement(instruction):
    return isinstance(instruction, pyquil.quil.Measurement)


def _get_qubit_index(qubit):
    if isinstance(qubit, int):
        return qubit

    if isinstance(qubit, pyquil.quilatom.Qubit):
        return qubit.index


def _qubits_to_wires(qubits, qubit_to_wire_map):
    if isinstance(qubits, Sequence):
        return [qubit_to_wire_map[_get_qubit_index(qubit)] for qubit in qubits]

    return qubit_to_wire_map[_get_qubit_index(qubits)]


class ParametrizedGate:
    def __init__(self, pl_gate, pyquil_qubits, pyquil_params, is_inverted):
        self.pl_gate = pl_gate
        self.pyquil_qubits = pyquil_qubits
        self.pyquil_params = pyquil_params
        self.is_inverted = is_inverted

    def instantiate(self, variable_map, qubit_to_wire_map):
        resolved_params = _resolve_params(self.pyquil_params, variable_map)
        resolved_wires = _qubits_to_wires(self.pyquil_qubits, qubit_to_wire_map)

        pl_gate_instance = self.pl_gate(*resolved_params, wires=resolved_wires)

        if self.is_inverted:
            pl_gate_instance.inv()

        return pl_gate_instance


class ParametrizedQubitUnitary:
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self, wires):
        return qml.QubitUnitary(self.matrix, wires=wires)


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
            instruction for instruction in self.program.instructions if _is_declaration(instruction)
        ]

    def _load_measurements(self):
        self._measurements = [
            instruction for instruction in self.program.instructions if _is_measurement(instruction)
        ]

        self._measurement_variable_names = set(
            [measurement.classical_reg.name for measurement in self._measurements]
        )

    def _is_defined_gate(self, gate):
        return gate.name in self._defined_gate_names

    def _is_measurement_variable(self, memory_placeholder):
        return memory_placeholder in self._measurements

    def _load_qubit_to_wire_map(self, wires):
        if len(wires) != len(self.qubits):
            raise qml.DeviceError(
                "The number of given wires does not match the number of qubits in the PyQuil Program. "
                + "{} wires were given, Program has {} qubits".format(len(wires), len(self.qubits))
            )

        self._qubit_to_wire_map = dict(zip(self.qubits, wires))

        return self._qubit_to_wire_map

    def _resolve_gate_matrix(self, gate):
        gate_matrix = self._matrix_dictionary[gate.name]

        for i, modifier in enumerate(gate.modifiers):
            if modifier == "CONTROLLED":
                gate_matrix = _controlled_matrix(gate_matrix)

        return gate_matrix

    def _check_variable_map(self, variable_map):
        for declaration in self._declarations:
            if not declaration.name in variable_map:
                # If the variable is used in measurement we don't complain
                if not declaration.name in self._measurement_variable_names:
                    raise qml.DeviceError(
                        (
                            "The PyQuil program defines a variable {} that is not present in the given variable map. "
                            + "Instruction: {}"
                        ).format(declaration.name, declaration)
                    )

    def __init__(self, program):
        self.program = program
        self.qubits = program.get_qubits()

        self._load_defined_gate_names()
        self._load_declarations()
        self._load_measurements()
        self._load_template()

    @property
    def defined_gates(self):
        return self.program.defined_gates

    @property
    def defined_gate_names(self):
        return self._defined_gate_names

    @property
    def declarations(self):
        return self._declarations

    @property
    def defined_variable_names(self):
        return [declaration.name for declaration in self._declarations]

    def _load_template(self):
        self._parametrized_gates = []

        for i, instruction in enumerate(self.program.instructions):
            # Skip all statements that are not gates (RESET, MEASURE, PRAGMA, ...)
            if not _is_gate(instruction):
                if not _is_declaration(instruction):
                    warnings.warn(
                        "Instruction Nr. {} is not supported by PennyLane and was ignored: {}".format(
                            i + 1, instruction
                        )
                    )

                continue

            # Rename for better readability
            gate = instruction

            if _is_forked(gate):
                raise qml.DeviceError(
                    "Forked gates can not be imported into PennyLane, as this functionality is not supported. "
                    + "Instruction Nr. {}, {} was a forked gate.".format(i + 1, gate)
                )

            resolved_gate = _resolve_gate(gate)

            # If the gate is a DefGate or not all CONTROLLED statements can be resolved
            # we resort to QubitUnitary
            if _is_controlled(resolved_gate) or self._is_defined_gate(resolved_gate):
                pl_gate = ParametrizedQubitUnitary(self._resolve_gate_matrix(resolved_gate))
            else:
                pl_gate = pyquil_inv_operation_map[resolved_gate.name]

            parametrized_gate = ParametrizedGate(
                pl_gate, gate.qubits, gate.params, _is_inverted(gate)
            )

            self._parametrized_gates.append(parametrized_gate)

    def template(self, variable_map={}, wires=None):
        if not wires:
            wires = range(len(self.qubits))

        variable_map = _normalize_variable_map(variable_map)

        qubit_to_wire_map = self._load_qubit_to_wire_map(wires)
        self._check_variable_map(variable_map)

        for parametrized_gate in self._parametrized_gates:
            parametrized_gate.instantiate(variable_map, qubit_to_wire_map)

    def __call__(self, variable_map={}, wires=None):
        self.template(variable_map, wires)

    def __str__(self):
        return "PennyLane Program Loader for PyQuil Program:\n" + str(self.program)


def load_program(program: pyquil.Program):
    """Load a pyquil.Program instance as a PennyLane template.

    During loading, gates are converted to PennyLane gates as far as possible. If
    the gates are not supported they are replaced with QubitUnitary instances. The
    import ignores all statements that are not declarations or gates (e.g. pragmas,
    classical control flow and measurements).

    Every variable that is present in the Program and that is not used as the target
    register of a measurement has to be provided in the `variable_map` of the template. 
    
    Args:
        program (pyquil.Program): The program that should be loaded
    
    Returns:
        ProgramLoader: a ProgramLoader instance that can be called like a template
    """

    return ProgramLoader(program)


def load_quil(quil_str: str):
    """Load a quil string as a PennyLane template.

    During loading, gates are converted to PennyLane gates as far as possible. If
    the gates are not supported they are replaced with QubitUnitary instances. The
    import ignores all statements that are not declarations or gates (e.g. pragmas,
    classical control flow and measurements).

    Every variable that is present in the Program and that is not used as the target
    register of a measurement has to be provided in the `variable_map` of the template. 
    
    Args:
        quil_str (str): The program that should be loaded
    
    Returns:
        ProgramLoader: a ProgramLoader instance that can be called like a template
    """

    return load_program(pyquil.Program(quil_str))


def load_quil_from_file(file_path: str):
    """Load a quil file as a PennyLane template.

    During loading, gates are converted to PennyLane gates as far as possible. If
    the gates are not supported they are replaced with QubitUnitary instances. The
    import ignores all statements that are not declarations or gates (e.g. pragmas,
    classical control flow and measurements).

    Every variable that is present in the Program and that is not used as the target
    register of a measurement has to be provided in the `variable_map` of the template. 
    
    Args:
        file_path (str): The path to the quil file that should be loaded
    
    Returns:
        ProgramLoader: a ProgramLoader instance that can be called like a template
    """

    with open(file_path, "r") as file:
        quil_str = file.read()

    return load_quil(quil_str)
