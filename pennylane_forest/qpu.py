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
from pyquil.paulis import sI, sX, sY, sZ
from pyquil.operator_estimation import ExperimentSetting, TensorProductState, TomographyExperiment, measure_observables, group_experiments
from pyquil.quilbase import Gate


class QPUDevice(ForestDevice):
    r"""Forest QPU device for PennyLane.

    Args:
        device (str): the name of the device to initialise.
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of observables.
        active_reset (bool): whether to actively reset qubits instead of waiting for
            for qubits to decay to the ground state naturally.
            Setting this to ``True`` results in a significantly faster expectation value
            evaluation when the number of shots is larger than ~1000.
        load_qc (bool): set to False to avoid getting the quantum computing
            device on initialization. This is convenient if not currently connected to the QPU.
        readout_error (list): specifies the conditional probabilities [p(0|0), p(1|1)], where
            p(i|j) denotes the prob of reading out i having sampled j; can be set to `None` if no
            readout errors need to be simulated; can only be set for the QPU-as-a-QVM
        symmetrize_readout (str): method to perform readout symmetrization, using exhaustive
            symmetrization by default
        calibrate_readout (str): method to perform calibration for readout error mitigation, normalizing
            by the expectation value in the +1-eigenstate of the observable by default

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
    name = "Forest QPU Device"
    short_name = "forest.qpu"
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hermitian"}

    def __init__(self, device, *, shots=1024, active_reset=True, load_qc=True, readout_error=None,
                 symmetrize_readout="exhaustive", calibrate_readout="plus-eig", **kwargs):

        if readout_error is not None and load_qc:
            raise ValueError("Readout error cannot be set on the physical QPU")

        self.readout_error = readout_error

        self._eigs = {}

        if "wires" in kwargs:
            raise ValueError("QPU device does not support a wires parameter.")

        if shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")

        aspen_match = re.match(r"Aspen-\d+-([\d]+)Q", device)
        num_wires = int(aspen_match.groups()[0])

        super().__init__(num_wires, shots, **kwargs)

        if load_qc:
            self.qc = get_qc(device, as_qvm=False, connection=self.connection)
        else:
            self.qc = get_qc(device, as_qvm=True, connection=self.connection)

        self.active_reset = active_reset
        self.symmetrize_readout = symmetrize_readout
        self.calibrate_readout = calibrate_readout

    def pre_measure(self):
        # pass
        ## code below borrowed from `qvm.py`
        """Run the QVM"""
        # pylint: disable=attribute-defined-outside-init
        for e in self.obs_queue:
            wires = e.wires

            if e.name in ["PauliX", "PauliY", "PauliZ", "Identity", "Hadamard"]:
                pass

            elif e.name == "Hermitian":
                # For arbitrary Hermitian matrix H, let U be the unitary matrix
                # that diagonalises it, and w_i be the eigenvalues.
                H = e.parameters[0]
                Hkey = tuple(H.flatten().tolist())

                if Hkey in self._eigs:
                    # retrieve eigenvectors
                    U = self._eigs[Hkey]["eigvec"]
                else:
                    # store the eigenvalues corresponding to H
                    # in a dictionary, so that they do not need to
                    # be calculated later
                    w, U = np.linalg.eigh(H)
                    self._eigs[Hkey] = {"eigval": w, "eigvec": U}

                # Perform a change of basis before measuring by applying U^ to the circuit
                self.apply("QubitUnitary", wires, [U.conj().T])

        prag = Program(Pragma("INITIAL_REWIRING", ['"PARTIAL"']))

        if self.active_reset:
            prag += RESET()

        self.prog = prag + self.prog

        qubits = list(self.prog.get_qubits())
        ro = self.prog.declare("ro", "BIT", len(qubits))
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

    def expval(self, observable, wires, par):
        # identify Experiment Settings for each of the possible observables
        d_expt_settings = {
            "Identity": [ExperimentSetting(TensorProductState(), sI(0))],
            "PauliX": [ExperimentSetting(TensorProductState(), sX(0))],
            "PauliY": [ExperimentSetting(TensorProductState(), sY(0))],
            "PauliZ": [ExperimentSetting(TensorProductState(), sZ(0))],
            "Hadamard": [ExperimentSetting(TensorProductState(), float(np.sqrt(1/2)) * sX(0)),
                         ExperimentSetting(TensorProductState(), float(np.sqrt(1/2)) * sZ(0))]
        }
        # expectation values for single-qubit observables
        if len(wires) == 1:

            if observable in ["PauliX", "PauliY", "PauliZ", "Identity", "Hadamard"]:
                qubit = self.qc.qubits()[0]
                prep_prog = Program([instr for instr in self.program if isinstance(instr, Gate)])
                if self.readout_error is not None:
                    prep_prog.define_noisy_readout(qubit, p00=self.readout_error[0],
                                                          p11=self.readout_error[1])
                tomo_expt = TomographyExperiment(settings=d_expt_settings[observable], program=prep_prog)
                grouped_tomo_expt = group_experiments(tomo_expt)
                meas_obs = list(measure_observables(self.qc, grouped_tomo_expt,
                                                    active_reset=self.active_reset,
                                                    symmetrize_readout=self.symmetrize_readout,
                                                    calibrate_readout=self.calibrate_readout))
                return np.sum([expt_result.expectation for expt_result in meas_obs])

            elif observable == 'Hermitian':
                # <H> = \sum_i w_i p_i
                Hkey = tuple(par[0].flatten().tolist())
                w = self._eigs[Hkey]['eigval']
                return w[0]*p0 + w[1]*p1

            else:
                raise ValueError("Unknown observable")

        # Multi-qubit observable
        # ----------------------
        # Currently, we only support qml.expval.Hermitian(A, wires),
        # where A is a 2^N x 2^N matrix acting on N wires.
        #
        # Eventually, we will also support tensor products of Pauli
        # matrices in the PennyLane UI.

        probs = self.probabilities(wires)

        if expectation == 'Hermitian':
            Hkey = tuple(par[0].flatten().tolist())
            w = self._eigs[Hkey]['eigval']
            # <A> = \sum_i w_i p_i
            return w @ probs

    def probabilities(self, wires):
        """Returns the (marginal) probabilities of the quantum state.

        Args:
            wires (Sequence[int]): sequence of wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            array: array of shape ``[2**len(wires)]`` containing
            the probabilities of each computational basis state
        """

        # create an array of size [2^len(wires), 2] to store
        # the resulting probability of each computational basis state
        probs = np.zeros([2 ** len(wires), 2])
        probs[:, 0] = np.arange(2 ** len(wires))

        # extract the measured samples
        res = np.array([self.state[w] for w in wires]).T
        for i in res:
            # for each sample, calculate which
            # computational basis state it corresponds to
            cb = np.sum(2 ** np.arange(len(wires) - 1, -1, -1) * i)
            # add a tally for this computational basis state
            # to our array of basis probabilities
            probs[cb, 1] += 1

        # sort the probabilities by the first column,
        # and divide by the number of shots
        probs = probs[probs[:, 0].argsort()] / self.shots
        probs = probs[:, 1]

        return probs
