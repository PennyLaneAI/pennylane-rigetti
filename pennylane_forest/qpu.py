"""
QPU Device
==========

**Module name:** :mod:`pennylane_forest.qpu`

.. currentmodule:: pennylane_forest.qpu

This module contains the :class:`~.QPUDevice` class, a PennyLane device that allows
evaluation and differentiation of Rigetti's Forest Quantum Processing Units (QPUs)
using PennyLane.

Classes
-------

.. autosummary::
   QPUDevice

Code details
~~~~~~~~~~~~
"""
import re

from pyquil import get_qc

# from .qvm import QVMDevice
from .device import ForestDevice
from ._version import __version__

import numpy as np

from pyquil import get_qc
from pyquil.api._quantum_computer import _get_qvm_with_topology
from pyquil.gates import MEASURE, RESET
from pyquil.quil import Pragma, Program
from pyquil.paulis import sX, sY, sZ
from pyquil.operator_estimation import ExperimentSetting, TensorProductState, TomographyExperiment, measure_observables, group_experiments
from pyquil.quilbase import Gate


class QPUDevice(ForestDevice):
    r"""Forest QPU device for PennyLane.

    Args:
        device (str): the name of the device to initialise.
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of expectations.
        active_reset (bool): whether to actively reset qubits instead of waiting for
            for qubits to decay to the ground state naturally.
            Setting this to ``True`` results in a significantly faster expectation value
            evaluation when the number of shots is larger than ~1000.
        load_qc (bool): set to False to avoid getting the quantum computing
            device on initialization. This is convenient if not currently connected to the QPU.

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
    name = 'Forest QPU Device'
    short_name = 'forest.qpu'
    expectations = {'PauliX', 'PauliY', 'PauliZ', 'Identity', 'Hadamard', 'Hermitian'}

    def __init__(self, device, *, shots=1024, active_reset=False, load_qc=True, **kwargs):
        self._eigs = {}

        if 'wires' in kwargs:
            raise ValueError("QPU device does not support a wires parameter.")

        if shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")

        aspen_match = re.match(r'Aspen-\d+-([\d]+)Q', device)
        num_wires = int(aspen_match.groups()[0])

        # super(QVMDevice, self).__init__(num_wires, shots, **kwargs) #pylint: disable=bad-super-call
        super().__init__(num_wires, shots, **kwargs)

        if load_qc:
            self.qc = get_qc(device, as_qvm=False, connection=self.connection)
        else:
            self.qc = get_qc(device, as_qvm=True, connection=self.connection)

        self.active_reset = active_reset

    def pre_expval(self):
        # pass
        ## code below borrowed from `qvm.py`
        """Run the QVM"""
        # pylint: disable=attribute-defined-outside-init
        for e in self.expval_queue:
            wires = e.wires

            if e.name == 'PauliX':
                # X = H.Z.H
                self.apply('Hadamard', wires, [])

            elif e.name == 'PauliY':
                # Y = (HS^)^.Z.(HS^) and S^=SZ
                self.apply('PauliZ', wires, [])
                self.apply('S', wires, [])
                self.apply('Hadamard', wires, [])

            elif e.name == 'Hadamard':
                # H = Ry(-pi/4)^.Z.Ry(-pi/4)
                self.apply('RY', wires, [-np.pi/4])

            elif e.name == 'Hermitian':
                # For arbitrary Hermitian matrix H, let U be the unitary matrix
                # that diagonalises it, and w_i be the eigenvalues.
                H = e.parameters[0]
                Hkey = tuple(H.flatten().tolist())

                if Hkey in self._eigs:
                    # retrieve eigenvectors
                    U = self._eigs[Hkey]['eigvec']
                else:
                    # store the eigenvalues corresponding to H
                    # in a dictionary, so that they do not need to
                    # be calculated later
                    w, U = np.linalg.eigh(H)
                    self._eigs[Hkey] = {'eigval': w, 'eigvec': U}

                # Perform a change of basis before measuring by applying U^ to the circuit
                self.apply('QubitUnitary', wires, [U.conj().T])

        prag = Program(Pragma('INITIAL_REWIRING', ['"PARTIAL"']))

        if self.active_reset:
            prag += RESET()

        self.prog = prag + self.prog

        qubits = list(self.prog.get_qubits())
        ro = self.prog.declare('ro', 'BIT', len(qubits))
        for i, q in enumerate(qubits):
            self.prog.inst(MEASURE(q, ro[i]))

        self.prog.wrap_in_numshots_loop(self.shots)

        if "pyqvm" in self.qc.name:
            bitstring_array = self.qc.run(self.prog)
        else:
            executable = self.qc.compile(self.prog)
            bitstring_array = self.qc.run(executable=executable)

        self.state = {}
        for i, q in enumerate(qubits):
            self.state[q] = bitstring_array[:, i]

    def expval(self, expectation, wires, par):
        d_expectation = {'PauliZ': sZ}
        if len(wires) == 1:
            qubit = self.qc.qubits()[0]
            prep_prog = Program([instr for instr in self.program if isinstance(instr, Gate)])
            expt_setting = ExperimentSetting(TensorProductState(), d_expectation[expectation](qubit))
            tomo_expt = TomographyExperiment(settings=[expt_setting], program=prep_prog)
            grouped_tomo_expt = group_experiments(tomo_expt)
            meas_obs = list(measure_observables(self.qc, grouped_tomo_expt))
            return np.sum(expt_result.expectation for expt_result in meas_obs)

        # if len(wires) == 1:
        #     # 1 qubit observable
        #     evZ = np.mean(1-2*self.state[wires[0]])

        #     # for single qubit state probabilities |psi|^2 = (p0, p1),
        #     # we know that p0+p1=1 and that <Z>=p0-p1
        #     p0 = (1+evZ)/2
        #     p1 = (1-evZ)/2

        #     if expectation == 'Identity':
        #         # <I> = \sum_i p_i
        #         return p0 + p1

        #     if expectation == 'Hermitian':
        #         # <H> = \sum_i w_i p_i
        #         Hkey = tuple(par[0].flatten().tolist())
        #         w = self._eigs[Hkey]['eigval']
        #         return w[0]*p0 + w[1]*p1

        #     return evZ

        # # Multi-qubit observable
        # # ----------------------
        # # Currently, we only support qml.expval.Hermitian(A, wires),
        # # where A is a 2^N x 2^N matrix acting on N wires.
        # #
        # # Eventually, we will also support tensor products of Pauli
        # # matrices in the PennyLane UI.

        # probs = self.probabilities(wires)

        # if expectation == 'Hermitian':
        #     Hkey = tuple(par[0].flatten().tolist())
        #     w = self._eigs[Hkey]['eigval']
        #     # <A> = \sum_i w_i p_i
        #     return w @ probs
