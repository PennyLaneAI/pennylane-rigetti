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
from pyquil.numpy_simulator import NumpyWavefunctionSimulator

from .device import ForestDevice
from .wavefunction import observable_map, spectral_decomposition_qubit
from ._version import __version__


class NumpyWavefunctionDevice(ForestDevice):
    r"""NumpyWavefunction simulator device for PennyLane.

    Args:
        wires (int): the number of qubits to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
    """
    name = 'pyQVM NumpyWavefunction Simulator Device'
    short_name = 'forest.numpy_wavefunction'

    observables = {'PauliX', 'PauliY', 'PauliZ', 'Hadamard', 'Hermitian', 'Identity'}

    def __init__(self, wires, *, shots=0, **kwargs):
        super().__init__(wires, shots, **kwargs)
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
        #pennylane_forest/wavefunction.py
        self.state = self.qc.execute(self.prog).wf_simulator.wf.flatten()

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
