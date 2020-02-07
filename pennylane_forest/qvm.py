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
from pennylane.variable import Variable

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
            * ``9q-qvm-square``: a :math:`9\times 9` lattice.
            * ``Nq-pyqvm`` or ``9q-pyqvm-square``, for the same as the above but run
              via the built-in pyQuil pyQVM device.
            * Any other supported Rigetti device architecture.
            * Graph topology representing the device architecture.

        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of observables.
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

    def __init__(self, device, *, shots=1024, noisy=False, **kwargs):

        if shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")

        # ignore any 'wires' keyword argument passed to the device
        kwargs.pop("wires", None)
        analytic = kwargs.get("analytic", False)
        timeout = kwargs.pop("timeout", None)

        self._compiled_program = None
        """Union[None, pyquil.ExecutableDesignator]: the latest compiled program. If parametric
        compilation is turned on, this will be a parametric program."""

        self.parametric_compilation = kwargs.get("parametric_compilation", True)

        if self.parametric_compilation:
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

        if analytic:
            raise ValueError("QVM device cannot be run in analytic=True mode.")

        # get the number of wires
        if isinstance(device, nx.Graph):
            # load a QVM based on a graph topology
            num_wires = device.number_of_nodes()
        elif isinstance(device, str):
            # the device string must match a valid QVM device, i.e.
            # N-qvm, or 9q-square-qvm, or Aspen-1-16Q-A
            wire_match = re.search(r"(\d+)(q|Q)", device)

            if wire_match is None:
                # with the current Rigetti naming scheme, this error should never
                # appear as long as the QVM quantum computer has the correct name
                raise ValueError("QVM device string does not indicate the number of qubits!")

            num_wires = int(wire_match.groups()[0])
        else:
            raise ValueError(
                "Required argument device must be a string corresponding to "
                "a valid QVM quantum computer, or a NetworkX graph object."
            )

        super().__init__(num_wires, shots, analytic=analytic, **kwargs)

        # get the qc
        if isinstance(device, nx.Graph):
            self.qc = _get_qvm_with_topology(
                "device", topology=device, noisy=noisy, connection=self.connection
            )
        elif isinstance(device, str):
            self.qc = get_qc(device, as_qvm=True, noisy=noisy, connection=self.connection)

        if timeout is not None:
            self.qc.compiler.client.timeout = timeout

        self.wiring = {i: q for i, q in enumerate(self.qc.qubits())}
        self.active_reset = False

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
            wires = self.remap_wires(operation.wires)

            if i > 0 and operation.name in ("QubitStateVector", "BasisState"):
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been applied "
                    "on a {} device.".format(operation.name, self.short_name)
                )

            # Prepare for parametric compilation
            par = []
            for param in operation.params:
                if isinstance(param, Variable):
                    # Using the idx for each Variable instance to specify the
                    # corresponding symbolic parameter
                    parameter_string = "theta" + str(param.idx)

                    if parameter_string not in self._parameter_reference_map:
                        # Create a new PyQuil memory reference and store it in the
                        # parameter reference map if it was not done so already
                        current_ref = self.prog.declare(parameter_string, "REAL")
                        self._parameter_reference_map[parameter_string] = current_ref

                    # Store the numeric value bound to the symbolic parameter
                    self._parameter_map[parameter_string] = [param.val]

                    # Appending the parameter reference to the parameters
                    # of the corresponding operation
                    par.append(self._parameter_reference_map[parameter_string])
                else:
                    par.append(param)

            self.prog += self._operation_map[operation.name](*par, *wires)

        self.prog += self.apply_rotations(rotations)

    def generate_samples(self):
        if "pyqvm" in self.qc.name:
            return self.qc.run(self.prog, memory_map=self._parameter_map)

        if self.circuit_hash is None or not self.parametric_compilation:
            # No hash provided or parametric compilation was set to False
            # Compile the program
            self._compiled_program = self.qc.compile(self.prog)
            return self.qc.run(executable=self._compiled_program)

        if self.circuit_hash not in self._compiled_program_dict:
            # Compiling this specific program for the first time
            # Store the compiled program with the corresponding hash
            self._compiled_program_dict[self.circuit_hash] = self.qc.compile(self.prog)

        # The program has been compiled, store as the latest compiled program
        self._compiled_program = self._compiled_program_dict[self.circuit_hash]
        samples = self.qc.run(executable=self._compiled_program, memory_map=self._parameter_map)
        return samples

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
            self._parameter_map = {}
            self._parameter_reference_map = {}
