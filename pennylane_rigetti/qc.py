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
from collections import OrderedDict

from pyquil import Program
from pyquil.api import QAMExecutionResult, QuantumComputer, QuantumExecutable
from pyquil.gates import RESET, MEASURE
from pyquil.quil import Pragma

from pennylane import DeviceError, numpy as np
from pennylane.wires import Wires

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
         wires (Iterable[Number, str]): Iterable that contains unique labels for the
            qubits as numbers or strings (i.e., ``['q1', ..., 'qN']``).
            The number of labels must match the number of qubits accessible on the backend.
            If not provided, qubits are addressed as consecutive integers ``[0, 1, ...]``, and their number
            is inferred from the backend.
        active_reset (bool): whether to actively reset qubits instead of waiting for
            for qubits to decay to the ground state naturally.
            Setting this to ``True`` results in a significantly faster expectation value
            evaluation when the number of shots is larger than ~1000.
        parallel (bool): If set to ``True`` batched circuits are executed in parallel.
        max_threads (int): If ``parallel`` is set to True, this controls the maximum number of threads
            that can be used during parallel execution of jobs. Has no effect if ``parallel`` is False.

    Keyword args:
        compiler_timeout (int): number of seconds to wait for a response from quilc (default 10).
        execution_timeout (int): number of seconds to wait for a response from the QVM (default 10).
        parametric_compilation (bool): a boolean value of whether or not to use parametric
            compilation.
    """
    version = __version__
    author = "Rigetti Computing Inc."

    def __init__(
        self,
        device,
        *,
        shots=1000,
        wires=None,
        active_reset=False,
        parallel=False,
        max_threads=4,
        **kwargs,
    ):
        if shots is not None and shots <= 0:
            raise ValueError("Number of shots must be a positive integer or None.")

        if parallel and max_threads <= 0:
            raise ValueError("max_threads must be set to a positive integer greater than 0")

        self._parallel = parallel
        self._max_threads = max_threads

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

            self._batched_parameter_map = {}
            """dict[str, list[float]]: a map of broadcasted parameter names to each of their values"""

            self._batch_size = 0
            """the batch size of the currently executing circuit"""

        timeout_args = self._get_timeout_args(**kwargs)

        self.qc = self.get_qc(device, **timeout_args)

        self.num_wires = len(self.qc.qubits())

        if wires is None:
            # infer the number of modes from the device specs
            # and use consecutive integer wire labels
            wires = range(self.num_wires)

        if isinstance(wires, int):
            raise ValueError(
                f"Device has a fixed number of {self.num_wires} qubits. The wires argument can only be used "
                "to specify an iterable of wire labels."
            )

        if self.num_wires != len(wires):
            raise ValueError(
                f"Device has a fixed number of {self.num_wires} qubits and "
                f"cannot be created with {len(wires)} wires."
            )

        self.wiring = dict(enumerate(self.qc.qubits()))
        self.active_reset = active_reset

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

    def define_wire_map(self, wires):
        if hasattr(self, "wiring"):
            device_wires = Wires(self.wiring)
        else:
            # if no wiring given, use consecutive wire labels
            device_wires = Wires(range(self.num_wires))

        return OrderedDict(zip(wires, device_wires))

    def apply(self, operations, **kwargs):
        """Applies the given quantum operations."""
        prag = Program(Pragma("INITIAL_REWIRING", ['"PARTIAL"']))
        if self.active_reset:
            prag += RESET()
        self.prog = prag + self.prog

        if self.parametric_compilation:
            self.prog += self.apply_parametric_operations(operations)
        else:
            self.prog += self.apply_circuit_operations(operations)

        rotations = kwargs.get("rotations", [])
        self.prog += self.apply_rotations(rotations)

        qubits = sorted(self.wiring.values())
        ro = self.prog.declare("ro", "BIT", len(qubits))
        for i, q in enumerate(qubits):
            self.prog.inst(MEASURE(q, ro[i]))

        self.prog.wrap_in_numshots_loop(self.shots)

    def apply_parametric_operations(self, operations):
        """Applies a parametric program by applying parametric operation with symbolic parameters.

        Args:
            operations (List[pennylane.Operation]): quantum operations that need to be applied

        Returns:
            pyquil.Prgram(): a pyQuil Program with the given operations
        """
        prog = Program()
        # Apply the circuit operations
        for i, operation in enumerate(operations):
            # map the operation wires to the physical device qubits
            device_wires = self.map_wires(operation.wires)

            if i > 0 and operation.name in ("QubitStateVector", "StatePrep", "BasisState"):
                raise DeviceError(
                    f"Operation {operation.name} cannot be used after other Operations have already been applied "
                    f"on a {self.short_name} device."
                )

            # Prepare for parametric compilation
            par = []
            if operation.batch_size is not None:
                parameter_string = f"theta{i}"
                if parameter_string not in self._parameter_reference_map:
                    # Create a new PyQuil memory reference and store it in the
                    # parameter reference map if it was not done so already
                    current_ref = self.prog.declare(parameter_string, "REAL")
                    self._parameter_reference_map[parameter_string] = current_ref

                # Store the values bound to the symbolic parameter
                self._batched_parameter_map[parameter_string] = operation.data[0]

                # Appending the parameter reference to the parameters
                # of the corresponding operation
                par.append(self._parameter_reference_map[parameter_string])
            else:
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

            prog += self._operation_map[operation.name](*par, *device_wires.labels)

        return prog

    @classmethod
    def capabilities(cls):
        """Get the capabilities of this device class.
        Returns:
            dict[str->*]: results
        """
        capabilities = super().capabilities().copy()
        capabilities.update(supports_broadcasting=True)
        return capabilities

    def compile(self) -> QuantumExecutable:
        """Compiles the program for the target device"""
        return self.qc.compile(self.prog)

    def _compile_with_cache(self, circuit_hash=None) -> QuantumExecutable:
        """When parametric compilation is enabled, fetches the compiled program from the cache if it exists.
        If not, the program is compiled and stored in the cache.
        If parametric compilation is disabled, this just compiles the program
        Args:
            circuit_hash: The circuit hash to use to fetch or store the compiled program. If ``None``, uses
            the circuit_hash from the currently running circuit.
        """
        if not self.parametric_compilation:
            return self.compile()

        if circuit_hash is None:
            circuit_hash = self.circuit_hash

        compiled_program = self._compiled_program_dict.get(circuit_hash, None)
        if compiled_program is None:
            compiled_program = self.compile()
            self._compiled_program_dict[circuit_hash] = compiled_program
        return compiled_program

    def execute(self, circuit, **kwargs):
        """Executes the given circuit"""
        if self.parametric_compilation:
            self._batch_size = circuit.batch_size
            self._circuit_hash = circuit.graph.hash
        return super().execute(circuit, **kwargs)

    def batch_execute(self, circuits):
        """Execute a batch of quantum circuits on the device.
        The circuits are represented by tapes, and they are executed one-by-one using the
        device's ``execute`` method. The results are collected in a list.
        Args:
            circuits (list[~.tape.QuantumTape]): circuits to execute on the device
        Returns:
            list[array[float]]: list of measured value(s)
        """
        if not self._parallel or len(circuits) <= 1:
            return super().batch_execute(circuits)

        tasks = []

        for circuit in circuits:
            self.reset()

            self.apply(circuit.operations, rotations=circuit.diagonalizing_gates)

            program = self.prog.copy()

            parameters = None
            if self.parametric_compilation:
                parameters = self._parameter_map
                if circuit.batch_size is not None:
                    for i in range(circuit.batch_size):
                        for region, values in self._batched_parameter_map.items():
                            parameters[region] = values[i]
                        tasks.append((program, parameters, circuit.graph.hash))
                else:
                    tasks.append((program, parameters, circuit.graph.hash))

            else:
                tasks.append((program, parameters, circuit.graph_hash))

        pool_size = len(tasks)
        if self._max_threads is not None:
            pool_size = min(self._max_threads, pool_size)

        with ThreadPool(pool_size) as pool:
            results = pool.starmap(self._run_task, tasks)

        # Batched circuits get broken down into individual tasks to maximize parallelism.
        # This step joins those tasks into a single result array so we return the expected
        # result shape for those circuits.
        normalized_results = []
        i = 0
        for circuit in circuits:
            if circuit.batch_size is None:
                normalized_results.append(results[i])
                i += 1
            else:
                normalized_results.append(np.array(results[i : i + circuit.batch_size]))
                i += circuit.batch_size

        return normalized_results

    def _run_task(self, program, parameters, circuit_hash):
        program = self._compile_with_cache(circuit_hash=circuit_hash)
        if self.parametric_compilation:
            for region, value in parameters.items():
                program.write_memory(region_name=region, value=value)
        results = self.qc.run(program)
        return self.extract_samples(results)

    def generate_samples(self):
        """Executes the program on the QuantumComputer and uses the results to return the
        computational basis samples of all wires."""
        self._compiled_program = self._compile_with_cache()
        if self.parametric_compilation:
            results = []
            # Set the parameter values in executable memory
            for region, value in self._parameter_map.items():
                self._compiled_program.write_memory(region_name=region, value=value)
            if self._batch_size:
                for i in range(self._batch_size):
                    for region, values in self._batched_parameter_map.items():
                        self._compiled_program.write_memory(region_name=region, value=values[i])
                    samples = self.qc.run(self._compiled_program)
                    samples = self.extract_samples(samples)
                    results.append(samples)
                return np.array(results)

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
            self._batched_parameter_map = {}
            self._batch_size = None
