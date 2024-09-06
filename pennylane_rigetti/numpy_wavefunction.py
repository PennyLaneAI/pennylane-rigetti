"""
NumpyWavefunction simulator device
==================================

**Module name:** :mod:`pennylane_rigetti.numpy_wavefunction`

.. currentmodule:: pennylane_rigetti.numpy_wavefunction

This module contains the :class:`~.NumpyWavefunctionDevice` class, a PennyLane device that allows
evaluation and differentiation of pyQuil's NumpyWavefunctionSimulator using PennyLane.


Classes
-------

.. autosummary::
   NumpyWavefunctionDevice

Code details
~~~~~~~~~~~~
"""

from pyquil.pyqvm import PyQVM
from pyquil.simulation import NumpyWavefunctionSimulator

from .device import RigettiDevice
from ._version import __version__


class NumpyWavefunctionDevice(RigettiDevice):
    r"""NumpyWavefunction simulator device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
    """

    name = "pyQVM NumpyWavefunction Simulator Device"
    short_name = "rigetti.numpy_wavefunction"

    observables = {"PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian", "Identity", "Prod"}

    def __init__(self, wires, *, shots=None):
        super().__init__(wires, shots)
        self.qc = PyQVM(n_qubits=len(self.wires), quantum_simulator_type=NumpyWavefunctionSimulator)
        self._state = None

    @classmethod
    def capabilities(cls):  # pylint: disable=missing-function-docstring
        capabilities = super().capabilities().copy()
        capabilities.update(
            returns_state=True,
        )
        return capabilities

    @property
    def state(self):  # pylint: disable=missing-function-docstring
        return self._state

    def apply(self, operations, **kwargs):
        self.reset()
        self.qc.wf_simulator.reset()
        super().apply(operations, **kwargs)

        # TODO: currently, the PyQVM considers qubit 0 as the leftmost bit and therefore
        # returns amplitudes in the opposite of the Rigetti Lisp QVM (which considers qubit
        # 0 as the rightmost bit). This may change in the future, so in the future this
        # might need to get udpated to be similar to the pre_measure function of
        # pennylane_rigetti/wavefunction.py
        self._state = self.qc.execute(self.prog).wf_simulator.wf.flatten()
