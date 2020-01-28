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
import uuid
import abc

import numpy as np

from pyquil import Program
from pyquil.api._base_connection import ForestConnection
from pyquil.api._config import PyquilConfig

from pyquil.quil import DefGate
from pyquil.gates import X, Y, Z, H, PHASE, RX, RY, RZ, CZ, SWAP, CNOT

# following gates are not supported by PennyLane
from pyquil.gates import S, T, CPHASE00, CPHASE01, CPHASE10, CPHASE, CCNOT, CSWAP, ISWAP, PSWAP

from pennylane import QubitDevice

from ._version import __version__


pyquil_config = PyquilConfig()


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
    return [X(w) for w, p in zip(wires, par) if p == 1]


def qubit_unitary(par, *wires):
    r"""Define a pyQuil custom unitary quantum operation.

    Args:
        par (array): a :math:`2^N\times 2^N` unitary matrix
            representing a custom quantum operation.
        wires (list): list of wires to prepare the basis state on

    Returns:
        list: list of PauliX matrix operators acting on each wire
    """
    if par.shape[0] != par.shape[1]:
        raise ValueError("Qubit unitary must be a square matrix.")

    if not np.allclose(par @ par.conj().T, np.identity(par.shape[0])):
        raise ValueError("Qubit unitary matrix must be unitary.")

    if par.shape != tuple([2 ** len(wires)] * 2):
        raise ValueError("Qubit unitary matrix must be 2^Nx2^N, where N is the number of wires.")

    # Get the Quil definition for the new gate
    gate_definition = DefGate("U_{}".format(str(uuid.uuid4())[:8]), par)
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


class ForestDevice(QubitDevice):
    r"""Abstract Forest device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.

    Keyword args:
        forest_url (str): the Forest URL server. Can also be set by
            the environment variable ``FOREST_URL``, or in the ``~/.qcs_config``
            configuration file. Default value is ``"https://forest-server.qcs.rigetti.com"``.
        qvm_url (str): the QVM server URL. Can also be set by the environment
            variable ``QVM_URL``, or in the ``~/.forest_config`` configuration file.
            Default value is ``"http://127.0.0.1:5000"``.
        quilc_url (str): the compiler server URL. Can also be set by the environment
            variable ``QUILC_URL``, or in the ``~/.forest_config`` configuration file.
            Default value is ``"http://127.0.0.1:6000"``.
    """
    pennylane_requires = ">=0.6"
    version = __version__
    author = "Josh Izaac"

    _operation_map = pyquil_operation_map
    _capabilities = {"model": "qubit", "tensor_observables": True}

    def __init__(self, wires, shots=1000, analytic=False,  **kwargs):
        super().__init__(wires, shots, analytic=analytic)
        self.analytic = analytic
        self.forest_url = kwargs.get("forest_url", pyquil_config.forest_url)
        self.qvm_url = kwargs.get("qvm_url", pyquil_config.qvm_url)
        self.compiler_url = kwargs.get("compiler_url", pyquil_config.quilc_url)

        self.connection = ForestConnection(
            sync_endpoint=self.qvm_url,
            compiler_endpoint=self.compiler_url,
            forest_cloud_endpoint=self.forest_url,
        )

        # The following environment variables are deprecated I think

        # api_key (str): the Forest API key. Can also be set by the environment
        #     variable ``FOREST_API_KEY``, or in the ``~/.qcs_config`` configuration file.
        # user_id (str): the Forest user ID. Can also be set by the environment
        #     variable ``FOREST_USER_ID``, or in the ``~/.qcs_config`` configuration file.
        # qpu_url (str): the QPU server URL. Can also be set by the environment
        #     variable ``QPU_URL``, or in the ``~/.forest_config`` configuration file.

        # if 'api_key' in kwargs:
        #     os.environ['FOREST_API_KEY'] = kwargs['api_key']

        # if 'user_id' in kwargs:
        #     os.environ['FOREST_USER_ID'] = kwargs['user_id']

        # if 'qpu_url' in kwargs:
        #     os.environ['QPU_URL'] = kwargs['qpu_url']

        self.reset()

    @property
    def program(self):
        """View the last evaluated Quil program"""
        return self.prog

    def remap_wires(self, wires):
        """Use the wiring specified for the device if applicable.

        Returns:
            list: wires as integers corresponding to the wiring if applicable
        """
        if hasattr(self, "wiring"):
            return [int(self.wiring[i]) for i in wires]

        return [int(w) for w in wires]

    def apply(self, operations, **kwargs):
        # pylint: disable=attribute-defined-outside-init
        rotations = kwargs.get("rotations", [])

        # Storing the active wires
        self._active_wires = ForestDevice.active_wires(operations + rotations)

        # Apply the circuit operations
        for i, operation in enumerate(operations):
            # number of wires on device
            wires = self.remap_wires(operation.wires)
            par = operation.parameters

            if i > 0 and operation.name in ("QubitStateVector", "BasisState"):
                raise DeviceError("Operation {} cannot be used after other Operations have already been applied "
                                  "on a {} device.".format(operation.name, self.short_name))

            self.prog += self._operation_map[operation.name](*par, *wires)

        # Apply the circuit rotations
        for operation in rotations:
            wires = self.remap_wires(operation.wires)
            par = operation.parameters
            self.prog += self._operation_map[operation.name](*par, *wires)

    def reset(self):
        self.prog = Program()
        self._active_wires = set()
        self._state = None

    @property
    def operations(self):
        return set(self._operation_map.keys())

    def mat_vec_product(self, mat, vec, wires):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        Args:
            mat (array): matrix to multiply
            vec (array): state vector to multiply
            wires (Sequence[int]): target subsystems

        Returns:
            array: output vector after applying ``mat`` to input ``vec`` on specified subsystems
        """
        num_wires = len(wires)

        if mat.shape != (2 ** num_wires, 2 ** num_wires):
            raise ValueError(
                f"Please specify a {2**num_wires} x {2**num_wires} matrix for {num_wires} wires."
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
        mat = np.reshape(mat, [2] * len(wires) * 2)

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
        axes = (np.arange(len(wires), 2 * len(wires)), wires)

        # After the tensor dot operation, the resulting array
        # will have shape ``size=[2, 2, ..., 2]``,
        # where ``len(size) == num_wires``, corresponding
        # to a valid state of the system.
        tdot = np.tensordot(mat, vec, axes=axes)

        # Tensordot causes the axes given in `wires` to end up in the first positions
        # of the resulting tensor. This corresponds to a (partial) transpose of
        # the correct output state
        # We'll need to invert this permutation to put the indices in the correct place
        unused_idxs = [idx for idx in range(self.num_wires) if idx not in wires]
        perm = wires + unused_idxs

        # argsort gives the inverse permutation
        inv_perm = np.argsort(perm)
        state_multi_index = np.transpose(tdot, inv_perm)

        return np.reshape(state_multi_index, 2 ** self.num_wires)

    def probability(self, wires=None):
        """Return the (marginal) probability of each computational basis
        state from the last run of the device.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.

        .. warning:: This method will have to be redefined for hardware devices, since it uses
            the ``device._state`` attribute. This attribute might not be available for such devices.

        Args:
            wires (Sequence[int]): Sequence of wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            List[float]: list of the probabilities
        """
        if self._state is None:
            return None

        wires = wires or range(self.num_wires)
        wires = self.remap_wires(wires)
        prob = self.marginal_prob(np.abs(self._state) ** 2, wires)
        return prob
