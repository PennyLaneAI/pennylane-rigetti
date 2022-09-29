"""
QVM Device
==========

**Module name:** :mod:`pennylane_forest.qvm`

.. currentmodule:: pennylane_forest.qvm

This module contains the :class:`~.QVMDevice` class, a PennyLane device that allows
evaluation and differentiation of Rigetti's Forest Quantum Virtual Machines (QVMs)
using PennyLane.

Classes
-------

.. autosummary::
   QVMDevice

Code details
~~~~~~~~~~~~
"""
import re

import networkx as nx
from pyquil import get_qc
from pyquil.api._quantum_computer import _get_qvm_with_topology
from pyquil.gates import MEASURE, RESET
from pyquil.quil import Pragma, Program

from pennylane import DeviceError

from ._version import __version__
from .device import ForestDevice


class QVMDevice(ForestDevice):
    r"""Forest QVM device for PennyLane.

    This device supports both the Rigetti Lisp QVM, as well as the built-in pyQuil pyQVM.
    If using the pyQVM, the ``qvm_url`` QVM server url keyword argument does not need to
    be set.

    Args:
        device (Union[str, nx.Graph]): the name or topology of the device to initialise.

            * ``Nq-qvm``: for a fully connected/unrestricted N-qubit QVM
            * ``9q-square-qvm``: a :math:`9\times 9` lattice.
            * ``Nq-pyqvm`` or ``9q-square-pyqvm``, for the same as the above but run
              via the built-in pyQuil pyQVM device.
            * Any other supported Rigetti device architecture.
            * Graph topology representing the device architecture.

        shots (None, int, list[int]): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. If ``None``, the device calculates probability, expectation values,
            and variances analytically. If an integer, it specifies the number of samples to estimate these quantities.
            If a list of integers is passed, the circuit evaluations are batched over the list of shots.
        wires (Iterable[Number, str]): Iterable that contains unique labels for the
            qubits as numbers or strings (i.e., ``['q1', ..., 'qN']``).
            The number of labels must match the number of qubits accessible on the backend.
            If not provided, qubits are addressed as consecutive integers [0, 1, ...], and their number
            is inferred from the backend.
        noisy (bool): set to ``True`` to add noise models to your QVM.

    Keyword args:
        forest_url (str): the Forest URL server. Can also be set by
            the environment variable ``FOREST_SERVER_URL``, or in the ``~/.qcs_config``
            configuration file. Default value is ``"https://forest-server.qcs.rigetti.com"``.
        qvm_url (str): the QVM server URL. Can also be set by the environment
            variable ``QVM_URL``, or in the ``~/.forest_config`` configuration file.
            Default value is ``"http://127.0.0.1:5000"``.
        compiler_url (str): the compiler server URL. Can also be set by the environment
            variable ``COMPILER_URL``, or in the ``~/.forest_config`` configuration file.
            Default value is ``"http://127.0.0.1:6000"``.
        timeout (int): number of seconds to wait for a response from the client.
        parametric_compilation (bool): a boolean value of whether or not to use parametric
            compilation.
    """
    name = "Forest QVM Device"
    short_name = "forest.qvm"
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hermitian"}

    def __init__(self, device, *, wires=None, shots=1000, noisy=False, **kwargs):

        if shots is not None and shots <= 0:
            raise ValueError("Number of shots must be a positive integer or None.")

        timeout = kwargs.pop("timeout", None)

        self._compiled_program = None
        """Union[None, pyquil.ExecutableDesignator]: the latest compiled program. If parametric
        compilation is turned on, this will be a parametric program."""

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

        if shots is None:
            raise ValueError("QVM device cannot be used for analytic computations.")

        self.client_configuration = super()._get_client_configuration()

        # get the qc
        if isinstance(device, nx.Graph):
            self.qc = _get_qvm_with_topology(
                name="device", topology=device, noisy=noisy, client_configuration=self.client_configuration, qvm_type="qvm", compiler_timeout=5.0, execution_timeout=5.0,
            )
        elif isinstance(device, str):
            self.qc = get_qc(device, as_qvm=True, noisy=noisy)

        self.num_wires = len(self.qc.qubits())

        if wires is None:
            # infer the number of modes from the device specs
            # and use consecutive integer wire labels
            wires = range(self.num_wires)

        if isinstance(wires, int):
            raise ValueError(
                "Device has a fixed number of {} qubits. The wires argument can only be used "
                "to specify an iterable of wire labels.".format(self.num_wires)
            )

        if self.num_wires != len(wires):
            raise ValueError(
                "Device has a fixed number of {} qubits and "
                "cannot be created with {} wires.".format(self.num_wires, len(wires))
            )

        super().__init__(wires, shots, **kwargs)

        if timeout is not None:
            self.qc.compiler.client.timeout = timeout

        self.wiring = {i: q for i, q in enumerate(self.qc.qubits())}
        self.active_reset = False

    def execute(self, circuit, **kwargs):

        if self.parametric_compilation:
            self._circuit_hash = circuit.graph.hash

        return super().execute(circuit, **kwargs)

    def apply(self, operations, **kwargs):
        """Run the QVM"""
        # pylint: disable=attribute-defined-outside-init
        if self.parametric_compilation and "pyqvm" not in self.qc.name:
            self.apply_parametric_program(operations, **kwargs)
        else:
            super().apply(operations, **kwargs)

        prag = Program(Pragma("INITIAL_REWIRING", ['"PARTIAL"']))

        if self.active_reset:
            prag += RESET()

        self.prog = prag + self.prog

        qubits = sorted(self.wiring.values())
        ro = self.prog.declare("ro", "BIT", len(qubits))
        for i, q in enumerate(qubits):
            self.prog.inst(MEASURE(q, ro[i]))

        self.prog.wrap_in_numshots_loop(self.shots)

    def apply_parametric_program(self, operations, **kwargs):
        """Applies a parametric program by applying parametric
        operation with symbolic parameters.
        """
        # pylint: disable=attribute-defined-outside-init
        rotations = kwargs.get("rotations", [])

        # Storing the active wires
        self._active_wires = ForestDevice.active_wires(operations + rotations)

        # Apply the circuit operations
        for i, operation in enumerate(operations):
            # map the operation wires to the physical device qubits
            device_wires = self.map_wires(operation.wires)

            if i > 0 and operation.name in ("QubitStateVector", "BasisState"):
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been applied "
                    "on a {} device.".format(operation.name, self.short_name)
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

            self.prog += self._operation_map[operation.name](*par, *device_wires.labels)

        self.prog += self.apply_rotations(rotations)

    def generate_samples(self):
        for region, value in self._parameter_map.items():
            self.prog.write_memory(region_name=region, value=value)

        if "pyqvm" in self.qc.name:
            return self.extract_samples(self.qc.run(self.prog))

        if self.circuit_hash is None:
            # Parametric compilation was set to False
            # Compile the program
            self._compiled_program = self.qc.compile(self.prog)
            return self.extract_samples(self.qc.run(executable=self._compiled_program))

        if self.circuit_hash not in self._compiled_program_dict:
            # Compiling this specific program for the first time
            # Store the compiled program with the corresponding hash
            self._compiled_program_dict[self.circuit_hash] = self.qc.compile(self.prog)

        # The program has been compiled, store as the latest compiled program
        self._compiled_program = self._compiled_program_dict[self.circuit_hash]
        return self.extract_samples(self.qc.run(executable=self._compiled_program))

    def extract_samples(self, execution_results):
        return execution_results.readout_data["ro"]

    @property
    def circuit_hash(self):
        if self.parametric_compilation:
            return self._circuit_hash

        return None

    @property
    def compiled_program(self):
        """Returns the latest program that was compiled for running.

        If parametric compilation is turned on, this will be a parametric program.

        The pyquil.ExecutableDesignator.program attribute stores the pyquil.Program
        instance. If no program was compiled yet, this property returns None.

        Returns:
            Union[None, pyquil.ExecutableDesignator]: the latest compiled program
        """
        return self._compiled_program

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
