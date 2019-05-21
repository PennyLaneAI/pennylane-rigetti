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
import numpy as np

from pyquil.pyqvm import PyQVM
from pyquil.numpy_simulator import NumpyWavefunctionSimulator

from .device import ForestDevice
from .wavefunction import expectation_map, spectral_decomposition_qubit
from ._version import __version__


class NumpyWavefunctionDevice(ForestDevice):
    r"""NumpyWavefunction simulator device for PennyLane.

    Args:
        wires (int): the number of qubits to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of expectations.
    """
    name = 'pyQVM NumpyWavefunction Simulator Device'
    short_name = 'forest.numpy_wavefunction'

    expectations = {'PauliX', 'PauliY', 'PauliZ', 'Hadamard', 'Hermitian', 'Identity'}

    def __init__(self, wires, *, shots=0, **kwargs):
        super().__init__(wires, shots, **kwargs)
        self.qc = PyQVM(n_qubits=wires, quantum_simulator_type=NumpyWavefunctionSimulator)
        self.state = None

    def pre_apply(self):
        self.reset()
        self.qc.wf_simulator.reset()

    def pre_expval(self):
        # TODO: currently, the PyQVM considers qubit 0 as the leftmost bit and therefore
        # returns amplitudes in the opposite of the Rigetti Lisp QVM (which considers qubit
        # 0 as the rightmost bit). This may change in the future, so in the future this
        # might need to get udpated to be similar to the pre_expval function of
        #pennylane_forest/wavefunction.py
        self.state = self.qc.execute(self.prog).wf_simulator.wf.flatten()

    def expval(self, expectation, wires, par):
        # measurement/expectation value <psi|A|psi>
        if expectation == 'Hermitian':
            A = par[0]
        else:
            A = expectation_map[expectation]

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

    def var(self, expectation, wires, par):
        # variance of the expectation value <psi|A|psi>
        if expectation == 'Hermitian':
            A = par[0]
        else:
            A = expectation_map[expectation]

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
        A = self.expand_one(A, wires)
        return np.vdot(self.state, A @ self.state).real

    def expand_one(self, U, wires):
        r"""Expand a one-qubit operator into a full system operator.

        Args:
          U (array): :math:`2\times 2` matrix
          wires (Sequence[int]): target subsystem

        Returns:
          array: :math:`2^n\times 2^n` matrix
        """
        if U.shape != (2, 2):
            raise ValueError('2x2 matrix required.')
        if len(wires) != 1:
            raise ValueError('One target subsystem required.')
        wires = wires[0]
        before = 2**wires
        after = 2**(self.num_wires-wires-1)
        U = np.kron(np.kron(np.eye(before), U), np.eye(after))
        return U
