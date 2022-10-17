"""
Base Forest device class
========================

**Module name:** :mod:`pennylane_forest.device`

.. currentmodule:: pennylane_forest.device

This module contains a base class for constructing Forest devices for PennyLane,
as well as some auxillary functions for converting PennyLane supported operations
(such as ``BasisState``, ``Rot``) to the equivalent pyQuil operations.

This class provides all the boilerplate for supporting Forest devices on PennyLane.

Auxiliary functions
-------------------

.. autosummary::
    basis_state
    rotation
    controlled_phase

Classes
-------

.. autosummary::
   ForestDevice

Code details
~~~~~~~~~~~~
"""
from abc import ABC, abstractmethod
from typing import Dict
import uuid

import numpy as np

from collections import OrderedDict

from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.quil import DefGate, Pragma
from pyquil.gates import (
    X,
    Y,
    Z,
    H,
    PHASE,
    RX,
    RY,
    RZ,
    CZ,
    SWAP,
    CNOT,
    S,
    T,
    CSWAP,
    I,
    CPHASE,
    CPHASE00,
    CPHASE01,
    CPHASE10,
    CCNOT,
    ISWAP,
    PSWAP,
    RESET,
    MEASURE
)
from qcs_api_client.client import QCSClientConfiguration

from pennylane import QubitDevice, DeviceError
from pennylane.wires import Wires

from ._version import __version__


def basis_state(par, *wires):
    """Decompose a basis state into a list of PauliX matrices.

    Args:
        par (array): an array of integers from the set {0,1} representing
            the computational basis state
        wires (list): list of wires to prepare the basis state on

    Returns:
        list: list of PauliX matrix operators acting on each wire
    """
    # pylint: disable=unused-argument
    # need the identity here because otherwise only the "p=1" wires register in the circuit
    return [X(w) if p == 1 else I(w) for w, p in zip(wires, par)]


def qubit_unitary(par, *wires):
    r"""Define a pyQuil custom unitary quantum operation.

    Args:
        par (array): a :math:`2^N\times 2^N` unitary matrix
            representing a custom quantum operation.
        wires (list): list of wires to prepare the basis state on

    Returns:
        list: list of PauliX matrix operators acting on each wire
    """
    # Get the Quil definition for the new gate
    u_str = str(uuid.uuid4())[:8]
    gate_definition = DefGate(f"U_{u_str}", par)
    # Get the gate constructor
    gate_constructor = gate_definition.get_constructor()
    return [gate_definition, gate_constructor(*wires)]


def rotation(a, b, c, wire):
    r"""Arbitrary one-qubit rotation using three Euler angles.

    Args:
        a, b, c (float): rotation angles
        wire (int): wire the rotation acts on

    Returns:
        list: Ry and Rz matrix operators acting on each wire
    """
    return [RZ(a, wire), RY(b, wire), RZ(c, wire)]


def controlled_phase(phi, q, *wires):
    r"""Maps the two-qubit controlled phase gate to the equivalent pyQuil command.

    Args:
        phi (float): the controlled phase angle
        q (int): an integer between 0 and 3 that corresponds to a state
            :math:`\{00, 01, 10, 11\}` on which the conditional phase
            gets applied
        wires (list): list of wires the CPHASE gate acts on

    Returns:
        pyquil.operation: the corresponding pyQuil operation
    """
    # pylint: disable=no-value-for-parameter
    if q == 0:
        return CPHASE00(phi, *wires)
    if q == 1:
        return CPHASE01(phi, *wires)
    if q == 2:
        return CPHASE10(phi, *wires)

    return CPHASE(phi, *wires)


# mapping operations supported by PennyLane to the
# corresponding pyQuil operation
pyquil_operation_map = {
    "BasisState": basis_state,
    "QubitUnitary": qubit_unitary,
    "PauliX": X,
    "PauliY": Y,
    "PauliZ": Z,
    "Hadamard": H,
    "CNOT": CNOT,
    "SWAP": SWAP,
    "CZ": CZ,
    "PhaseShift": PHASE,
    "RX": RX,
    "RY": RY,
    "RZ": RZ,
    "Rot": rotation,
    # the following gates are provided by the PL-Forest plugin
    "S": S,
    "T": T,
    "Toffoli": CCNOT,
    "CPHASE": controlled_phase,
    "CSWAP": CSWAP,
    "ISWAP": ISWAP,
    "PSWAP": PSWAP,
}


class ForestDevice(QubitDevice, ABC):
    r"""Abstract Forest device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.
    """
    pennylane_requires = ">=0.17"
    version = __version__
    author = "Rigetti Computing Inc."

    _operation_map = pyquil_operation_map
    _capabilities = {"model": "qubit", "tensor_observables": True}

    def __init__(self, device, *, wires=None, shots=1000, noisy=False, active_reset=False, **kwargs):
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

        self.qc = self.get_qc(device, noisy, **timeout_args)

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

        self.wiring = {i: q for i, q in enumerate(self.qc.qubits())}
        self.active_reset = active_reset

        super().__init__(wires, shots)
        self.reset()

    @abstractmethod
    def get_qc(self, device, noisy, **kwargs) -> QuantumComputer:
        pass

    @staticmethod
    def _get_client_configuration():
        return QCSClientConfiguration.load()

    @property
    def program(self):
        """View the last evaluated Quil program"""
        return self.prog

    @property
    def circuit_hash(self):
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

    def define_wire_map(self, wires):
        if hasattr(self, "wiring"):
            device_wires = Wires(self.wiring)
        else:
            # if no wiring given, use consecutive wire labels
            device_wires = Wires(range(self.num_wires))

        return OrderedDict(zip(wires, device_wires))

    def _get_timeout_args(self, **kwargs) -> Dict[str, float]:
        timeout_args = {}
        if "compiler_timeout" in kwargs:
            timeout_args["compiler_timeout"] = kwargs["compiler_timeout"]

        if "execution_timeout" in kwargs:
            timeout_args["execution_timeout"] = kwargs["execution_timeout"]

        return timeout_args

    def execute(self, circuit, **kwargs):
        if self.parametric_compilation:
            self._circuit_hash = circuit.graph.hash
        return super().execute(circuit, **kwargs)

    def apply(self, operations, **kwargs):
        prag = Program(Pragma("INITIAL_REWIRING", ['"PARTIAL"']))
        if self.active_reset:
            prag += RESET()
        self.prog = prag + self.prog

        if self.parametric_compilation and "pyqvm" not in self.qc.name:
            self.prog += self.apply_parametric_operations(operations)
        else:
            self.prog += self.apply_circuit_operations(operations)

        # pylint: disable=attribute-defined-outside-init
        rotations = kwargs.get("rotations", [])
        # Storing the active wires
        self._active_wires = ForestDevice.active_wires(operations + rotations)

        self.prog += self.apply_rotations(rotations)

        qubits = sorted(self.wiring.values())
        ro = self.prog.declare("ro", "BIT", len(qubits))
        for i, q in enumerate(qubits):
            self.prog.inst(MEASURE(q, ro[i]))

        self.prog.wrap_in_numshots_loop(self.shots)

    def apply_rotations(self, rotations):
        """Apply the circuit rotations.

        This method serves as an auxiliary method to :meth:`~.ForestDevice.apply`.

        Args:
            rotations (List[pennylane.Operation]): operations that rotate into the
                measurement basis

        Returns:
            pyquil.Program: the pyquil Program that specifies the corresponding rotations
        """
        rotation_operations = Program()
        for operation in rotations:
            # map the ops' wires to the wire labels used by the device
            device_wires = self.map_wires(operation.wires)
            par = operation.parameters
            rotation_operations += self._operation_map[operation.name](*par, *device_wires.labels)

        return rotation_operations

    def apply_circuit_operations(self, operations):
        """Apply circuit operations

        This method is an auxillary method to :meth:`~.ForestDevice.apply`

        Args:
            operations (List[pennylane.Operation]): quantum operations to apply to a program.

        Returns:
            pyquil.Program(): the pyQuil Program that has these operations applied
        """
        prog = Program()
        for i, operation in enumerate(operations):
            # map the ops' wires to the wire labels used by the device
            device_wires = self.map_wires(operation.wires)
            par = operation.parameters

            if isinstance(par, list) and par:
                if isinstance(par[0], np.ndarray) and par[0].shape == ():
                    # Array not supported
                    par = [float(i) for i in par]

            if i > 0 and operation.name in ("QubitStateVector", "BasisState"):
                name = operation.name
                short_name = self.short_name
                raise DeviceError(
                    f"Operation {name} cannot be used after other Operations have already "
                    f"been applied on a {short_name} device."
                )

            prog += self._operation_map[operation.name](*par, *device_wires.labels)

        return prog

    def apply_parametric_operations(self, operations):
        """Applies a parametric program by applying parametric
        operation with symbolic parameters.
        """
        prog = Program()
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

            prog += self._operation_map[operation.name](*par, *device_wires.labels)

        return prog

    def reset(self):
        self.prog = Program()
        self._active_wires = Wires([])
        self._state = None

    @property
    def operations(self):
        return set(self._operation_map.keys())

    def mat_vec_product(self, mat, vec, device_wire_labels):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        Args:
            mat (array): matrix to multiply
            vec (array): state vector to multiply
            device_wire_labels (Sequence[int]): labels of device subsystems

        Returns:
            array: output vector after applying ``mat`` to input ``vec`` on specified subsystems
        """
        num_wires = len(device_wire_labels)

        if mat.shape != (2**num_wires, 2**num_wires):
            raise ValueError(
                f"Please specify a {2 ** num_wires} x {2 ** num_wires} matrix for {num_wires} wires."
            )

        # first, we need to reshape both the matrix and vector
        # into blocks of 2x2 matrices, in order to do the higher
        # order matrix multiplication

        # Reshape the matrix to ``size=[2, 2, 2, ..., 2]``,
        # where ``len(size) == 2*len(wires)``
        #
        # The first half of the dimensions correspond to a
        # 'ket' acting on each wire/qubit, while the second
        # half of the dimensions correspond to a 'bra' acting
        # on each wire/qubit.
        #
        # E.g., if mat = \sum_{ijkl} (c_{ijkl} |ij><kl|),
        # and wires=[0, 1], then
        # the reshaped dimensions of mat are such that
        # mat[i, j, k, l] == c_{ijkl}.
        mat = np.reshape(mat, [2] * len(device_wire_labels) * 2)

        # Reshape the state vector to ``size=[2, 2, ..., 2]``,
        # where ``len(size) == num_wires``.
        # Each wire corresponds to a subsystem.
        #
        # E.g., if vec = \sum_{ijk}c_{ijk}|ijk>,
        # the reshaped dimensions of vec are such that
        # vec[i, j, k] == c_{ijk}.
        vec = np.reshape(vec, [2] * self.num_wires)

        # Calculate the axes on which the matrix multiplication
        # takes place. For the state vector, this simply
        # corresponds to the requested wires. For the matrix,
        # it is the latter half of the dimensions (the 'bra' dimensions).
        #
        # For example, if num_wires=3 and wires=[2, 0], then
        # axes=((2, 3), (2, 0)). This is equivalent to doing
        # np.einsum("ijkl,lnk", mat, vec).
        axes = (np.arange(len(device_wire_labels), 2 * len(device_wire_labels)), device_wire_labels)

        # After the tensor dot operation, the resulting array
        # will have shape ``size=[2, 2, ..., 2]``,
        # where ``len(size) == num_wires``, corresponding
        # to a valid state of the system.
        tdot = np.tensordot(mat, vec, axes=axes)

        # Tensordot causes the axes given in `wires` to end up in the first positions
        # of the resulting tensor. This corresponds to a (partial) transpose of
        # the correct output state
        # We'll need to invert this permutation to put the indices in the correct place
        unused_idxs = [idx for idx in range(self.num_wires) if idx not in device_wire_labels]
        perm = device_wire_labels + unused_idxs

        # argsort gives the inverse permutation
        inv_perm = np.argsort(perm)
        state_multi_index = np.transpose(tdot, inv_perm)

        return np.reshape(state_multi_index, 2**self.num_wires)

    def analytic_probability(self, wires=None):
        """Return the (marginal) probability of each computational basis
        state from the last run of the device.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.

        .. warning:: This method will have to be redefined for hardware devices, since it uses
            the ``device._state`` attribute. This attribute might not be available for such devices.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            List[float]: list of the probabilities
        """
        if self._state is None:
            return None

        prob = self.marginal_prob(np.abs(self._state) ** 2, wires)
        return prob
