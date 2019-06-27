"""
Wavefunction simulator device
=============================

**Module name:** :mod:`pennylane_forest.wavefunction`

.. currentmodule:: pennylane_forest.wavefunction

This module contains the :class:`~.WavefunctionDevice` class, a PennyLane device that allows
evaluation and differentiation of Rigetti's WavefunctionSimulator using PennyLane.


Auxiliary functions
-------------------

.. autosummary::
    spectral_decomposition_qubit


Classes
-------

.. autosummary::
   WavefunctionDevice

Code details
~~~~~~~~~~~~
"""
import itertools

import numpy as np

from pyquil.api import WavefunctionSimulator

from .device import ForestDevice
from ._version import __version__


I = np.identity(2)
X = np.array([[0, 1], [1, 0]]) #: Pauli-X matrix
Y = np.array([[0, -1j], [1j, 0]]) #: Pauli-Y matrix
Z = np.array([[1, 0], [0, -1]]) #: Pauli-Z matrix
H = np.array([[1, 1], [1, -1]])/np.sqrt(2) # Hadamard matrix


observable_map = {'PauliX': X, 'PauliY': Y, 'PauliZ': Z, 'Identity': I, 'Hadamard': H}


def spectral_decomposition_qubit(A):
    r"""Spectral decomposition of a :math:`2\times 2` Hermitian matrix.

    Args:
        A (array): :math:`2\times 2` Hermitian matrix

    Returns:
        (vector[float], list[array[complex]]): (a, P): eigenvalues and hermitian projectors
        such that :math:`A = \sum_k a_k P_k`.
    """
    d, v = np.linalg.eigh(A)
    P = []
    for k in range(2):
        temp = v[:, k]
        P.append(np.outer(temp, temp.conj()))
    return d, P


class WavefunctionDevice(ForestDevice):
    r"""Wavefunction simulator device for PennyLane.

    Args:
        wires (int): the number of qubits to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.

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
    """
    name = 'Forest Wavefunction Simulator Device'
    short_name = 'forest.wavefunction'

    observables = {'PauliX', 'PauliY', 'PauliZ', 'Hadamard', 'Hermitian', 'Identity'}

    def __init__(self, wires, *, shots=0, **kwargs):
        super().__init__(wires, shots, **kwargs)
        self.qc = WavefunctionSimulator(connection=self.connection)
        self.state = None

    def pre_measure(self):
        self.state = self.qc.wavefunction(self.prog).amplitudes

        # pyQuil uses the convention that the first qubit is the least significant
        # qubit. Here, we reverse this to make it the last qubit, matching PennyLane convention.
        self.state = self.state.reshape([2]*len(self.active_wires)).T.flatten()
        self.expand_state()

    def expand_state(self):
        """The pyQuil wavefunction simulator initializes qubits dymnically as they are requested.
        This method expands the state to the full number of wires in the device."""

        if len(self.active_wires) == self.num_wires:
            # all wires in the device have been initialised
            return

        # there are some wires in the device that have not yet been initialised
        inactive_wires = set(range(self.num_wires)) - self.active_wires

        # place the inactive subsystems in the vacuum state
        other_subsystems = np.zeros([2**len(inactive_wires)])
        other_subsystems[0] = 1

        # expand the state of the device into a length-num_wire state vector
        expanded_state = np.kron(self.state, other_subsystems).reshape([2]*self.num_wires)
        expanded_state = np.moveaxis(expanded_state, range(len(self.active_wires)), self.active_wires)
        expanded_state = expanded_state.flatten()

        self.state = expanded_state

    def expval(self, observable, wires, par):
        if observable == 'Hermitian':
            A = par[0]
        else:
            A = observable_map[observable]

        if self.shots == 0:
            # exact expectation value
            ev = self.ev(A, wires)
        else:
            # estimate the ev
            # sample Bernoulli distribution n_eval times / binomial distribution once
            a, P = spectral_decomposition_qubit(A)
            p0 = self.ev(P[0], wires)  # probability of measuring a[0]
            n0 = np.random.binomial(self.shots, p0)
            ev = (n0*a[0] +(self.shots-n0)*a[1]) / self.shots

        return ev

    def var(self, observable, wires, par):
        if observable == 'Hermitian':
            A = par[0]
        else:
            A = observable_map[observable]

        var = self.ev(A@A, wires) - self.ev(A, wires)**2
        return var

    def ev(self, A, wires):
        r"""Evaluates a one-qubit expectation in the current state.

        Args:
          A (array): :math:`2\times 2` Hermitian matrix corresponding to the expectation
          wires (Sequence[int]): target subsystem

        Returns:
          float: expectation value :math:`\left\langle{A}\right\rangle = \left\langle{\psi}\mid A\mid{\psi}\right\rangle`
        """
        # Expand the Hermitian observable over the entire subsystem
        A = self.expand(A, wires)
        return np.vdot(self.state, A @ self.state).real

    def expand(self, U, wires):
        r"""Expand a multi-qubit operator into a full system operator.

        Args:
            U (array): :math:`2^n \times 2^n` matrix where n = len(wires).
            wires (Sequence[int]): Target subsystems (order matters! the
                left-most Hilbert space is at index 0).

        Returns:
            array: :math:`2^N\times 2^N` matrix. The full system operator.
        """
        if self.num_wires == 1:
            # total number of wires is 1, simply return the matrix
            return U

        N = self.num_wires
        wires = np.asarray(wires)

        if np.any(wires < 0) or np.any(wires >= N) or len(set(wires)) != len(wires):
            raise ValueError("Invalid target subsystems provided in 'wires' argument.")

        if U.shape != (2**len(wires), 2**len(wires)):
            raise ValueError("Matrix parameter must be of size (2**len(wires), 2**len(wires))")

        # generate N qubit basis states via the cartesian product
        tuples = np.array(list(itertools.product([0, 1], repeat=N)))

        # wires not acted on by the operator
        inactive_wires = list(set(range(N))-set(wires))

        # expand U to act on the entire system
        U = np.kron(U, np.identity(2**len(inactive_wires)))

        # move active wires to beginning of the list of wires
        rearranged_wires = np.array(list(wires)+inactive_wires)

        # convert to computational basis
        # i.e., converting the list of basis state bit strings into
        # a list of decimal numbers that correspond to the computational
        # basis state. For example, [0, 1, 0, 1, 1] = 2^3+2^1+2^0 = 11.
        perm = np.ravel_multi_index(tuples[:, rearranged_wires].T, [2]*N)

        # permute U to take into account rearranged wires
        return U[:, perm][perm]
