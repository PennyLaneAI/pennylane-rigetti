import warnings
from collections.abc import Sequence

import numpy as np
import pennylane as qml
import pennylane_rigetti as plf
import pyquil
from pyquil import simulation
from pyquil.gates import Gate

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
    "S": qml.S,
    "T": qml.T,
    "CCNOT": qml.Toffoli,
    "CPHASE": lambda *params, wires: plf.ops.CPHASE(*params, 3, wires=wires),
    "CPHASE00": lambda *params, wires: plf.ops.CPHASE(*params, 0, wires=wires),
    "CPHASE01": lambda *params, wires: plf.ops.CPHASE(*params, 1, wires=wires),
    "CPHASE10": lambda *params, wires: plf.ops.CPHASE(*params, 2, wires=wires),
    "CSWAP": qml.CSWAP,
    "ISWAP": qml.ISWAP,
    "PSWAP": qml.PSWAP,
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


def _direct_sum(A, B):
    """Return the direct sums of two arrays.

    Args:
        A (np.array): The first array (upper left array)
        B (np.array): The second array (lower right array)

    Returns:
        np.array: The direct sum of the two input arrays
    """
    sum_matrix = np.zeros(np.add(A.shape, B.shape), dtype=A.dtype)
    sum_matrix[: A.shape[0], : A.shape[1]] = A
    sum_matrix[A.shape[0] :, A.shape[1] :] = B

    return sum_matrix


def _controlled_matrix(op):
    """Return the matrix associated with the controlled operation.

    Args:
        op (np.array): Array representing the operation
                       that should be controlled

    Returns:
        np.array: Array representing the controlled operations. If the input
        array has shape (N, N) the output shape is (2*N, 2*N).
    """
    return _direct_sum(np.eye(op.shape[0], dtype=op.dtype), op)


def _resolve_gate(gate):
    """Resolve the given pyquil Gate as far as possible.

    For example, the gate ``CONTROLLED CONTROLLED X`` will be resolved to ``CCNOT``.
    The gate ``CONTROLLED CONTROLLED RX(0.3)`` will be resolved to ``CONTROLLED CRX(0.3)``.

    Args:
        gate (pyquil.quil.Gate): The gate that should be resolved

    Returns:
        pyquil.quil.Gate: The maximally resolved gate
    """
    for i, modifier in enumerate(gate.modifiers):
        if modifier == "CONTROLLED":
            if gate.name in _control_map:
                stripped_gate = Gate(_control_map[gate.name], gate.params, gate.qubits)
                stripped_gate.modifiers = gate.modifiers.copy()
                del stripped_gate.modifiers[i]

                return _resolve_gate(stripped_gate)
            else:
                break

    return gate


def _resolve_params(params, parameter_map):
    """Resolve a parameter list with a given variable map.

    Args:
        params (List[Union[pyquil.quilatom.MemoryReference, object]]): The parameter list
        parameter_map (Dict[str, object]): The map that assigns values to variable names

    Returns:
        List[object]: The resolved parameters. This list does not contain MemoryReferences anymore.
    """
    resolved_params = []

    for param in params:
        if isinstance(param, pyquil.quilatom.MemoryReference):
            resolved_params.append(parameter_map[param.name])
        else:
            resolved_params.append(param)

    return resolved_params


def _normalize_parameter_map(parameter_map):
    """Normalize the given variable map.

    Variable maps can have keys that are either strings or
    pyquil.quil.MemoryReference instances. This methods replaces
    all MemoryReference instances with their name.

    Args:
        parameter_map (Dict[Union[str, pyquil.quil.MemoryReference], object]): The initial variable map.

    Returns:
        Dict[str, object]: Variable map with all MemoryReference instances replaced by their name
    """
    new_keys = list(parameter_map.keys())
    values = list(parameter_map.values())

    for i in range(len(new_keys)):
        if isinstance(new_keys[i], pyquil.quil.MemoryReference):
            new_keys[i] = new_keys[i].name

    return dict(zip(new_keys, values))


def _is_controlled(gate):
    """Determine if a gate is controlled.

    Args:
        gate (pyquil.quil.Gate): The gate that should be checked

    Returns:
        bool: True if the gate is controlled, False otherwise
    """
    return "CONTROLLED" in gate.modifiers


def _is_forked(gate):
    """Determine if a gate is forked.

    Args:
        gate (pyquil.quil.Gate): The gate that should be checked

    Returns:
        bool: True if the gate is forked, False otherwise
    """
    return "FORKED" in gate.modifiers


def _is_inverted(gate):
    """Determine if a gate is inverted.

    Args:
        gate (pyquil.quil.Gate): The gate that should be checked

    Returns:
        bool: True if the gate is inverted, False otherwise
    """
    return gate.modifiers.count("DAGGER") % 2 == 1


def _is_gate(instruction):
    """Determine if an instruction is a gate.

    Args:
        instruction (pyquil.quil.AbstractInstruction): The instruction that should be checked

    Returns:
        bool: True if the instruction is a gate, False otherwise
    """
    return isinstance(instruction, pyquil.quil.Gate)


def _is_declaration(instruction):
    """Determine if an instruction is a declaration.

    Args:
        instruction (pyquil.quil.AbstractInstruction): The instruction that should be checked

    Returns:
        bool: True if the instruction is a declaration, False otherwise
    """
    return isinstance(instruction, pyquil.quil.Declare)


def _is_measurement(instruction):
    """Determine if an instruction is a measurement.

    Args:
        instruction (pyquil.quil.AbstractInstruction): The instruction that should be checked

    Returns:
        bool: True if the instruction is a measurement, False otherwise
    """
    return isinstance(instruction, pyquil.quil.Measurement)


def _get_qubit_index(qubit):
    """Return the index of the given qubit.

    This function accepts qubit instances and integers and returns
    the integer index that is associated with the qubit.

    Args:
        qubit (Union[pyquil.quilatom.Qubit, int]): The qubit whose index shall be determined

    Returns:
        int: The qubit index
    """
    if isinstance(qubit, int):
        return qubit

    if isinstance(qubit, pyquil.quilatom.Qubit):
        return qubit.index


def _qubits_to_wires(qubits, qubit_to_wire_map):
    """Transform the given qubits with the given map between qubits and wires.

    Args:
        qubits (Union[Sequence[int], int]): The qubit(s) for which the wires shall be determined
        qubit_to_wire_map (Dict[int, int]): The map between qubits and wires

    Returns:
        Union[Sequence[int], int]: The wire(s) corresponding to the given qubit(s)
    """
    if isinstance(qubits, Sequence):
        return [qubit_to_wire_map[_get_qubit_index(qubit)] for qubit in qubits]

    return qubit_to_wire_map[_get_qubit_index(qubits)]


class ParametrizedGate:
    """Represent a parametrized PennyLane gate.

    Args:
        pl_gate (type): The PennyLane gate
        pyquil_qubits ([type]): The qubits the gate acts on in the corresponding pyquil Program
        pyquil_params ([type]): The gate's parameters in the corresponding pyquil Program
        is_inverted (bool): Indicates if the gate is inverted
    """

    def __init__(self, pl_gate, pyquil_qubits, pyquil_params, is_inverted):
        self.pl_gate = pl_gate
        self.pyquil_qubits = pyquil_qubits
        self.pyquil_params = pyquil_params
        self.is_inverted = is_inverted

    def instantiate(self, parameter_map, qubit_to_wire_map):
        """Instantiate the parametrized gate.

        Args:
            parameter_map (Dict[str, object]): The map that assigns values to variable names
            qubit_to_wire_map ([type]): The map that assigns wires to qubits

        Returns:
            qml.Operation: The initiated instance of the parametrized PennyLane gate
        """
        resolved_params = _resolve_params(self.pyquil_params, parameter_map)
        resolved_wires = _qubits_to_wires(self.pyquil_qubits, qubit_to_wire_map)

        pl_gate_instance = self.pl_gate(*resolved_params, wires=resolved_wires)

        if self.is_inverted:
            return qml.adjoint(pl_gate_instance)

        return pl_gate_instance


class ParametrizedQubitUnitary:
    """Represents a QubitUnitary instance already parametrized with a matrix.

    Args:
        matrix (np.array): The unitary gate matrix
    """

    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self, wires):
        """Instantiate the QubitUnitary on the given wires.

        Args:
            wires (List[int]): The wires the QubitUnitary acts on

        Returns:
            qml.QubitUnitary: The instantiate QubitUnitary instance
        """
        return qml.QubitUnitary(self.matrix, wires=wires)


class ProgramLoader:
    """Loads the given pyquil Program as a PennyLane template.

    The pyquil Program is parsed once at instantiation and can be
    applied as often as desired.

    Args:
        program (pyquil.quil.Program): The pyquil Program instance that should be loaded
    """

    _matrix_dictionary = simulation.matrices.QUANTUM_GATES

    def __init__(self, program):
        self.program = program
        self.qubits = program.get_qubits()

        self._load_defined_gate_names()
        self._load_declarations()
        self._load_measurements()
        self._load_template()

    def _load_defined_gate_names(self):
        """Extract the names of all defined gates of the pyquil Program."""
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
        """Extract the declarations of the pyquil Program."""
        self._declarations = [
            instruction for instruction in self.program.instructions if _is_declaration(instruction)
        ]

    def _load_measurements(self):
        """Extract the measurements of the pyquil Program."""
        measurements = [
            instruction for instruction in self.program.instructions if _is_measurement(instruction)
        ]

        self._measurement_variable_names = set(
            [measurement.classical_reg.name for measurement in measurements]
        )

    def _is_defined_gate(self, gate):
        """Determine if the given gate was defined in the pyquil Program.

        Args:
            gate (pyquil.quil.Gate): The gate that shall be checked

        Returns:
            bool: True if the gate is defined, False otherwise
        """
        return gate.name in self._defined_gate_names

    def _load_qubit_to_wire_map(self, wires):
        """Build the map that assigns wires to qubits.

        Args:
            wires (Sequence[int]): The wires that should be assigned to the qubits

        Raises:
            qml.DeviceError: When the number of given wires does not match the number of qubits in the pyquil Program

        Returns:
            Dict[int, int]: The map that assigns wires to qubits
        """
        if len(wires) != len(self.qubits):
            raise qml.DeviceError(
                "The number of given wires does not match the number of qubits in the PyQuil Program. "
                + "{} wires were given, Program has {} qubits".format(len(wires), len(self.qubits))
            )

        self._qubit_to_wire_map = dict(zip(self.qubits, wires))

        return self._qubit_to_wire_map

    def _resolve_gate_matrix(self, gate):
        """Resolve the matrix of the given pyquil gate.

        Args:
            gate (pyquil.quil.Gate): The gate whose matrix should be resolved

        Returns:
            np.array: The matrix of the given gate
        """
        gate_matrix = self._matrix_dictionary[gate.name]

        for i, modifier in enumerate(gate.modifiers):
            if modifier == "CONTROLLED":
                gate_matrix = _controlled_matrix(gate_matrix)

        return gate_matrix

    def _check_parameter_map(self, parameter_map):
        """Check that all variables of the program are defined.

        Only variables used in measurements need not be defined.

        Args:
            parameter_map (Dict[str, object]): Map that assigns values to variables

        Raises:
            qml.DeviceError: When not all variables are defined in the variable map
        """
        for declaration in self._declarations:
            if not declaration.name in parameter_map:
                # If the variable is used in measurement we don't complain
                if not declaration.name in self._measurement_variable_names:
                    raise qml.DeviceError(
                        (
                            "The PyQuil program defines a variable {} that is not present in the given variable map. "
                            + "Instruction: {}"
                        ).format(declaration.name, declaration)
                    )

    @property
    def defined_gates(self):
        """The custom gates defined in the pyquil Program.

        Returns:
            List[Union[pyquil.quil.DefGate, pyquil.quil.DefPermutationGate]]: The gates defined in the pyquil Program
        """
        return self.program.defined_gates

    @property
    def defined_qubits(self):
        """The qubit indices defined in the pyquil Program.

        Returns:
            List[int]: The qubit indices defined in the pyquil Program
        """
        return list(self.qubits)

    @property
    def defined_gate_names(self):
        """The names of the custom gates defined in the pyquil Program.

        Returns:
            List[str]: The names of the gates defined in the pyquil Program
        """
        return self._defined_gate_names

    @property
    def declarations(self):
        """The declarations in the pyquil Program.

        Returns:
            List[pyquil.quil.Declaration]: The declarations in the pyquil Program
        """
        return self._declarations

    @property
    def defined_variable_names(self):
        """The names of the variables defined in the pyquil Program

        Returns:
            List[str]: The names of the variables defined in the pyquil Program
        """
        return [declaration.name for declaration in self._declarations]

    def _load_template(self):
        """Load the template corresponding to the pyquil Program.

        Raises:
            qml.DeviceError: When the import of a forked gate is attempted
        """
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

    def template(self, wires, parameter_map=None):
        """Executes the template extracted from the pyquil Program.

        Args:
            wires (Sequence[int]): The wires on which the template shall be applied.
            parameter_map (Dict[Union[str, pyquil.quilatom.MemoryReference], object], optional): The
                map that assigns values to variables. Defaults to an empty dictionary.

        Returns:
            List[qml.Operation]: A list of all applied gates
        """
        if parameter_map is None:
            parameter_map = {}

        parameter_map = _normalize_parameter_map(parameter_map)

        qubit_to_wire_map = self._load_qubit_to_wire_map(wires)
        self._check_parameter_map(parameter_map)

        applied_gates = []
        for parametrized_gate in self._parametrized_gates:
            applied_gates.append(parametrized_gate.instantiate(parameter_map, qubit_to_wire_map))

        return applied_gates

    def __call__(self, wires, parameter_map=None):
        """Executes the template extracted from the pyquil Program.

        Args:
            wires (Sequence[int]): The wires on which the template shall be applied.
            parameter_map (Dict[Union[str, pyquil.quilatom.MemoryReference], object], optional): The
                map that assigns values to variables. Defaults to an empty dictionary.

        Returns:
            List[qml.Operation]: A list of all applied gates
        """
        return self.template(wires, parameter_map)

    def __str__(self):
        """Give the string representation of the ProgramLoader.

        Returns:
            str: The string representation of the ProgramLoader
        """
        return "PennyLane Program Loader for PyQuil Program:\n" + str(self.program)


def load_program(program: pyquil.Program):
    """Load a pyquil.Program instance as a PennyLane template.

    During loading, gates are converted to PennyLane gates as far as possible. If
    the gates are not supported they are replaced with QubitUnitary instances. The
    import ignores all statements that are not declarations or gates (e.g. pragmas,
    classical control flow and measurements).

    Every variable that is present in the Program and that is not used as the target
    register of a measurement has to be provided in the ``parameter_map`` of the template.

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
    register of a measurement has to be provided in the ``parameter_map`` of the template.

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
    register of a measurement has to be provided in the ``parameter_map`` of the template.

    Args:
        file_path (str): The path to the quil file that should be loaded

    Returns:
        ProgramLoader: a ProgramLoader instance that can be called like a template
    """

    with open(file_path, "r") as file:
        quil_str = file.read()

    return load_quil(quil_str)
