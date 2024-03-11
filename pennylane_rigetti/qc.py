"""
Base Quantum Computer device class
==================================

**Module name:** :mod:`pennylane_rigetti.qc`

.. currentmodule:: pennylane_rigetti.qc

This module contains the :class:`~.QuantumComputerDevice` base class that can be
used to build PennyLane devices from pyQuil QuantumComputers (such as a Rigetti Quantum Processor Unit).

Classes
-------

.. autosummary::
   QuantumComputerDevice

Code details
~~~~~~~~~~~~
"""

from abc import ABC, abstractmethod
from typing import Dict
from pennylane.utils import Iterable
from pennylane import DeviceError, numpy as np

from pyquil import Program
from pyquil.api import QPU, QAMExecutionResult, QuantumComputer, QuantumExecutable
from pyquil.gates import RESET, MEASURE
from pyquil.quil import Pragma

from .device import RigettiDevice
from ._version import __version__


class QuantumComputerDevice(RigettiDevice, ABC):
    r"""Abstract Quantum Computer device for PennyLane.

    This is a base class for common logic shared by pyQuil ``QuantumComputer``s (i.e. QVMs, QPUs).
    Children classes need to define a `get_qc` method that returns a pyQuil ``QuantumComputer``.

    Args:
        device (str): the name of the device to initialise.
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of observables.
        active_reset (bool): whether to actively reset qubits instead of waiting for
            for qubits to decay to the ground state naturally.
            Setting this to ``True`` results in a significantly faster expectation value
            evaluation when the number of shots is larger than ~1000.

    Keyword args:
        compiler_timeout (int): number of seconds to wait for a response from quilc (default 10).
        wires (Optional[Union[int, Iterable[Number, str]]]): Number of qubits represented by the device, or an iterable that contains
            unique labels for the qubits as numbers or strings (i.e., ``['q1', ..., 'qN']``). When an integer is provided, qubits
            are addressed as consecutive integers ``[0, 1, n]`` and mapped to the first available qubits on the device. `wires` must
            be provided for QPU devices. If not provided for a QVM device, then the number of wires is inferred from the QVM.
        execution_timeout (int): number of seconds to wait for a response from the QVM (default 10).
        parametric_compilation (bool): a boolean value of whether or not to use parametric
            compilation.
    """

    version = __version__
    author = "Rigetti Computing Inc."

    def __init__(self, device, *, wires=None, shots=1000, active_reset=False, **kwargs):
        if shots is not None and shots <= 0:
            raise ValueError("Number of shots must be a positive integer or None.")

        self._compiled_program = None

        self.parametric_compilation = kwargs.get("parametric_compilation", True)

        if self.parametric_compilation:
            self._circuit_hash = None
            """None or int: stores the hash of the circuit from the last execution which
            can be used for parametric compilation."""

            self._compiled_program_dict = {}
            """dict[int, pyquil.ExecutableDesignator]: stores circuit hashes associated
                with the corresponding compiled programs."""

            self._parameter_map = {}
            """dict[str, float]: stores the string of symbolic parameters associated with
                their numeric values. This map will be used to bind parameters in a parametric
                program using PyQuil."""

            self._parameter_reference_map = {}
            """dict[str, pyquil.quilatom.MemoryReference]: stores the string of symbolic
                parameters associated with their PyQuil memory references."""

        timeout_args = self._get_timeout_args(**kwargs)

        self.qc = self.get_qc(device, **timeout_args)

        qubits = sorted(
            [
                qubit.id
                for qubit in self.qc.quantum_processor.to_compiler_isa().qubits.values()
                if qubit.dead is False
            ]
        )

        if wires is None:
            if not isinstance(self.qc.qam, QPU):
                wires = len(qubits)
            else:
                raise ValueError("Wires must be specified for a QPU device. Got None for wires.")

        if isinstance(wires, int):
            if wires > len(qubits):
                raise ValueError(
                    "Wires must not exceed available qubits on the device. ",
                    f"The requested device only has {len(qubits)} qubits, received {wires}.",
                )
            wires = dict(zip(range(wires), qubits[:wires]))
        elif isinstance(wires, Iterable):
            if len(wires) > len(qubits):
                raise ValueError(
                    "Wires must not exceed available qubits on the device. ",
                    f"The requested device only has {len(qubits)} qubits, received {len(wires)}.",
                )
            wires = dict(zip(wires, qubits[: len(wires)]))
        else:
            raise ValueError(
                "Wires for a device must be an integer or an iterable of numbers or strings."
            )

        self.num_wires = len(qubits)
        self._qubits = qubits[: len(wires)]
        self.active_reset = active_reset
        self.wiring = wires

        super().__init__(wires, shots)

    @abstractmethod
    def get_qc(self, device, **kwargs) -> QuantumComputer:
        """Initializes and returns a pyQuil QuantumComputer that can run quantum programs"""

    @property
    def circuit_hash(self):
        """Returns the hash of the most recently executed circuit.

        If no circuit has been executed yet, or parametric compilation is disabled then None is returned.

        Returns:
            Union[str, None]: The circuit hash of the current
        """
        if self.parametric_compilation:
            return self._circuit_hash

        return None

    @property
    def compiled_program(self):
        """Returns the latest program that was compiled for running.

        If parametric compilation is turned on, this will be a parametric program.

        The program is returned as a string of the Quil code.
        If no program was compiled yet, this property returns None.

        Returns:
            Union[None, str]: the latest compiled program
        """
        return str(self._compiled_program) if self._compiled_program else None

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def program(self):
        """View the last evaluated Quil program"""
        return self.prog

    def _get_timeout_args(self, **kwargs) -> Dict[str, float]:
        timeout_args = {}
        if "compiler_timeout" in kwargs:
            timeout_args["compiler_timeout"] = kwargs["compiler_timeout"]

        if "execution_timeout" in kwargs:
            timeout_args["execution_timeout"] = kwargs["execution_timeout"]

        return timeout_args

    def apply(self, operations, **kwargs):
        """Applies the given quantum operations."""
        self.prog = Program(Pragma("INITIAL_REWIRING", ['"PARTIAL"']))
        if self.active_reset:
            self.prog += RESET()

        if self.parametric_compilation:
            self.prog += self.apply_parametric_operations(operations)
        else:
            self.prog += self.apply_circuit_operations(operations)

        rotations = kwargs.get("rotations", [])
        self.prog += self.apply_rotations(rotations)

        qubits = self._qubits
        ro = self.prog.declare("ro", "BIT", len(qubits))
        for i, q in enumerate(qubits):
            self.prog.inst(MEASURE(q, ro[i]))

        self.prog.wrap_in_numshots_loop(self.shots)

    def apply_parametric_operations(self, operations):
        """Applies a parametric program by applying parametric operation with symbolic parameters.

        Args:
            operations (List[pennylane.Operation]): quantum operations that need to be applied

        Returns:
            pyquil.Program(): a pyQuil Program with the given operations
        """
        prog = Program()
        # Apply the circuit operations
        for i, operation in enumerate(operations):
            # map the operation wires to the physical device qubits
            backend_wires = self.map_wires_to_backend(operation.wires)

            if i > 0 and operation.name in (
                "QubitStateVector",
                "StatePrep",
                "BasisState",
            ):
                raise DeviceError(
                    f"Operation {operation.name} cannot be used after other Operations have already been applied "
                    f"on a {self.short_name} device."
                )

            # Prepare for parametric compilation
            par = []
            for param in operation.data:
                if getattr(param, "requires_grad", False) and operation.name != "BasisState":
                    # Using the idx for trainable parameter objects to specify the
                    # corresponding symbolic parameter
                    parameter_string = "theta" + str(id(param))

                    if parameter_string not in self._parameter_reference_map:
                        # Create a new PyQuil memory reference and store it in the
                        # parameter reference map if it was not done so already
                        current_ref = self.prog.declare(parameter_string, "REAL")
                        self._parameter_reference_map[parameter_string] = current_ref

                    # Store the numeric value bound to the symbolic parameter
                    self._parameter_map[parameter_string] = [param.unwrap()]

                    # Appending the parameter reference to the parameters
                    # of the corresponding operation
                    par.append(self._parameter_reference_map[parameter_string])
                else:
                    par.append(param)

            prog += self._operation_map[operation.name](*par, *backend_wires.labels)

        return prog

    def compile(self) -> QuantumExecutable:
        """Compiles the program for the target device"""
        return self.qc.compile(self.prog)

    def execute(self, circuit, **kwargs):
        """Executes the given circuit"""
        if self.parametric_compilation:
            self._circuit_hash = circuit.graph.hash
        return super().execute(circuit, **kwargs)

    def generate_samples(self):
        """Executes the program on the QuantumComputer and uses the results to return the
        computational basis samples of all wires."""
        if self.parametric_compilation:
            # Set the parameter values in executable memory
            for region, value in self._parameter_map.items():
                self.prog.write_memory(region_name=region, value=value)
            # Fetch the compiled program, or compile and store it if it doesn't exist
            self._compiled_program = self._compiled_program_dict.get(self.circuit_hash, None)
            if self._compiled_program is None:
                self._compiled_program = self.compile()
                self._compiled_program_dict[self.circuit_hash] = self._compiled_program
        else:
            # Parametric compilation is disabled, just compile the program
            self._compiled_program = self.compile()

        results = self.qc.run(self._compiled_program)
        return self.extract_samples(results)

    def extract_samples(self, execution_results: QAMExecutionResult) -> np.ndarray:
        """Returns samples from the readout register on the execution results received after
        running a program on a pyQuil Quantum Abstract Machine.

        Returns:
            numpy.ndarray: Samples extracted from the readout register on the execution results.
        """
        return execution_results.readout_data.get("ro", [])

    def reset(self):
        """Resets the device after the previous run.

        Note:
            The ``_compiled_program`` and the ``_compiled_program_dict`` attributes are
            not reset such that these can be used upon multiple device execution.
        """
        super().reset()

        if self.parametric_compilation:
            self._circuit_hash = None
            self._parameter_map = {}
            self._parameter_reference_map = {}
