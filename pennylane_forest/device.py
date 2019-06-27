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
from pyquil.gates import (X, Y, Z, H, PHASE, RX, RY, RZ, CZ, SWAP, CNOT)
# following gates are not supported by PennyLane
from pyquil.gates import (S, T, CPHASE00, CPHASE01, CPHASE10, CPHASE, CCNOT, CSWAP, ISWAP, PSWAP)

from pennylane import Device

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
    return [X(w) for w, p in enumerate(par) if p == 1]


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

    if par.shape != tuple([2**len(wires)]*2):
        raise ValueError('Qubit unitary matrix must be 2^Nx2^N, where N is the number of wires.')

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
    'BasisState': basis_state,
    'QubitUnitary': qubit_unitary,
    "PauliX": X,
    "PauliY": Y,
    "PauliZ": Z,
    "Hadamard": H,
    'CNOT': CNOT,
    'SWAP': SWAP,
    'CZ': CZ,
    'PhaseShift': PHASE,
    'RX': RX,
    'RY': RY,
    'RZ': RZ,
    'Rot': rotation,
    # the following gates are provided by the PL-Forest plugin
    'S': S,
    'T': T,
    'CCNOT': CCNOT,
    'CPHASE': controlled_phase,
    'CSWAP': CSWAP,
    'ISWAP': ISWAP,
    'PSWAP': PSWAP,
}


class ForestDevice(Device):
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
    pennylane_requires = '>=0.4'
    version = __version__
    author = 'Josh Izaac'

    _operation_map = pyquil_operation_map

    def __init__(self, wires, shots, **kwargs):
        super().__init__(wires, shots)
        self.forest_url = kwargs.get('forest_url', pyquil_config.forest_url)
        self.qvm_url = kwargs.get('qvm_url', pyquil_config.qvm_url)
        self.compiler_url = kwargs.get('compiler_url', pyquil_config.quilc_url)

        self.connection = ForestConnection(
            sync_endpoint=self.qvm_url,
            compiler_endpoint=self.compiler_url,
            forest_cloud_endpoint=self.forest_url
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

    def apply(self, operation, wires, par):
        # pylint: disable=attribute-defined-outside-init
        self.prog += self._operation_map[operation](*par, *wires)

        # keep track of the active wires. This is required, as the
        # pyQuil wavefunction simulator creates qubits dynamically.
        if wires:
            self.active_wires = self.active_wires.union(set(wires))
        else:
            self.active_wires = set(range(self.num_wires))

    @abc.abstractmethod
    def pre_measure(self): #pragma no cover
        """Run the QVM or QPU"""
        raise NotImplementedError

    def reset(self):
        self.prog = Program()
        self.active_wires = set()
        self.state = None

    @property
    def operations(self):
        return set(self._operation_map.keys())
