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
from numpy.linalg import eigh
from pennylane.wires import Wires

from pyquil.api import WavefunctionSimulator

from .device import ForestDevice
from ._version import __version__


I = np.identity(2)
X = np.array([[0, 1], [1, 0]])  #: Pauli-X matrix
Y = np.array([[0, -1j], [1j, 0]])  #: Pauli-Y matrix
Z = np.array([[1, 0], [0, -1]])  #: Pauli-Z matrix
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard matrix


observable_map = {"PauliX": X, "PauliY": Y, "PauliZ": Z, "Identity": I, "Hadamard": H}


class WavefunctionDevice(ForestDevice):
    r"""Wavefunction simulator device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
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
    name = "Forest Wavefunction Simulator Device"
    short_name = "forest.wavefunction"

    observables = {"PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian", "Identity"}

    def __init__(self, wires, *, shots=None, **kwargs):
        super().__init__(wires, shots, **kwargs)
        self.connection = super()._get_connection(**kwargs)
        self.qc = WavefunctionSimulator(connection=self.connection)
        self._state = None

    def apply(self, operations, **kwargs):
        super().apply(operations, **kwargs)

        self._state = self.qc.wavefunction(self.prog).amplitudes

        # pyQuil uses the convention that the first qubit is the least significant
        # qubit. Here, we reverse this to make it the last qubit, matching PennyLane convention.
        self._state = self._state.reshape([2] * len(self._active_wires)).T.flatten()
        self.expand_state()

    @staticmethod
    def bit2dec(x):
        """Auxiliary method that converts a bitstring to a decimal integer
        using the PennyLane convention of bit ordering.

        Args:
            x (Iterable): bit string

        Returns:
            int: decimal value of the bitstring
        """
        y = 0
        for i, j in enumerate(x[::-1]):
            y += j << i
        return y

    def expand_state(self):
        """The pyQuil wavefunction simulator initializes qubits dymnically as they are requested.
        This method expands the state to the full number of wires in the device."""

        if len(self._active_wires) == self.num_wires:
            # all wires in the device have been initialised
            return

        # translate active wires to the device's labels
        device_active_wires = self.map_wires(self._active_wires)

        inactive_wires = [x for x in range(len(self.wires)) if x not in device_active_wires]

        # initialize the entire new expanded state to zeros
        expanded_state = np.zeros([2 ** len(self.wires)], dtype=self.C_DTYPE)

        # gather the bit strings for the subsystem made up of the active qubits
        subsystem_bit_strings = self.states_to_binary(
            np.arange(2 ** len(self._active_wires)), len(self._active_wires)
        )

        for string, amplitude in zip(subsystem_bit_strings, self._state):
            for w in inactive_wires:

                # expand the bitstring by inserting a zero bit for each inactive qubit
                string = np.insert(string, w, 0)

            # calculate the decimal value of the bit string, that gives the
            # index of the amplitude in the state vector
            decimal_val = self.bit2dec(string)
            expanded_state[decimal_val] = amplitude

        self._state = expanded_state
