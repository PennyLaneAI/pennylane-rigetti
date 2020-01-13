"""
NumpyWavefunction simulator device
==================================

**Module name:** :mod:`pennylane_forest.numpy_wavefunction`

.. currentmodule:: pennylane_forest.numpy_wavefunction

This module contains the :class:`~.NumpyWavefunctionDevice` class, a PennyLane device that allows
evaluation and differentiation of pyQuil's NumpyWavefunctionSimulator using PennyLane.


Classes
-------

.. autosummary::
   NumpyWavefunctionDevice

Code details
~~~~~~~~~~~~
"""
import itertools

import numpy as np

from pyquil.pyqvm import PyQVM
from pyquil.simulation import NumpyWavefunctionSimulator

from .wavefunction import WavefunctionDevice
from ._version import __version__


class NumpyWavefunctionDevice(WavefunctionDevice):
    r"""NumpyWavefunction simulator device for PennyLane.

    Args:
        wires (int): the number of qubits to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
    """
    name = "pyQVM NumpyWavefunction Simulator Device"
    short_name = "forest.numpy_wavefunction"

    observables = {"PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian", "Identity"}

    def __init__(self, wires, *, shots=1000, analytic=True, **kwargs):
        super(WavefunctionDevice, self).__init__(wires, shots, analytic, **kwargs)
        self.qc = PyQVM(n_qubits=wires, quantum_simulator_type=NumpyWavefunctionSimulator)
        self.state = None

    def pre_apply(self):
        self.reset()
        self.qc.wf_simulator.reset()

    def pre_measure(self):
        # TODO: currently, the PyQVM considers qubit 0 as the leftmost bit and therefore
        # returns amplitudes in the opposite of the Rigetti Lisp QVM (which considers qubit
        # 0 as the rightmost bit). This may change in the future, so in the future this
        # might need to get udpated to be similar to the pre_measure function of
        # pennylane_forest/wavefunction.py
        self.state = self.qc.execute(self.prog).wf_simulator.wf.flatten()
