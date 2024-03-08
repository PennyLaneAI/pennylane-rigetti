"""
Base Rigetti device class
========================

**Module name:** :mod:`pennylane_rigetti.device`

.. currentmodule:: pennylane_rigetti.device

This module contains a base class for constructing Rigetti devices for PennyLane,
as well as some auxillary functions for converting PennyLane supported operations
(such as ``BasisState``, ``Rot``) to the equivalent pyQuil operations.

This class provides all the boilerplate for supporting Rigetti devices on PennyLane.

Auxiliary functions
-------------------

.. autosummary::
    basis_state
    rotation
    controlled_phase

Classes
-------

.. autosummary::
   RigettiDevice

Code details
~~~~~~~~~~~~
"""

import uuid

import numpy as np

from collections import OrderedDict

from pyquil import Program
from pyquil.quil import DefGate
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
)

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
    "S": S,
    "T": T,
    "Toffoli": CCNOT,
    "CPHASE": controlled_phase,
    "CSWAP": CSWAP,
    "ISWAP": ISWAP,
    "PSWAP": PSWAP,
}


class RigettiDevice(QubitDevice):
    r"""Abstract Rigetti device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.
    """

    pennylane_requires = ">=0.18"
    version = __version__
    author = "Rigetti Computing Inc."

    _operation_map = pyquil_operation_map
    _capabilities = {"model": "qubit", "tensor_observables": True}

    def __init__(self, wires, shots=1000):
        super().__init__(wires, shots)
        self.prog = Program()
        self._state = None

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def program(self):
        """View the last evaluated Quil program"""
        return self.prog

    def define_wire_map(self, wires):
        if hasattr(self, "wiring"):
            device_wires = Wires(self.wiring)
        else:
            # if no wiring given, use consecutive wire labels
            device_wires = Wires(range(self.num_wires))

        return OrderedDict(zip(wires, device_wires))

    def apply(self, operations, **kwargs):
        self.prog += self.apply_circuit_operations(operations)
        self.prog += self.apply_rotations(kwargs.get("rotations", []))

    def apply_rotations(self, rotations):
        """Apply the circuit rotations.

        This method serves as an auxiliary method to :meth:`~.RigettiDevice.apply`.

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

        Args:
            operations (List[pennylane.Operation]): quantum operations that need to be applied

        Returns:
            pyquil.Program(): a pyQuil Program with the given operations
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

            if i > 0 and operation.name in ("QubitStateVector", "StatePrep", "BasisState"):
                name = operation.name
                short_name = self.short_name
                raise DeviceError(
                    f"Operation {name} cannot be used after other Operations have already "
                    f"been applied on a {short_name} device."
                )

            prog += self._operation_map[operation.name](*par, *device_wires.labels)

        return prog

    def reset(self):
        """Resets the device after the previous run."""
        self.prog = Program()
        self._state = None

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
