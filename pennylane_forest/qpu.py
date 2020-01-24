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

from .qvm import QVMDevice
from ._version import __version__

import numpy as np

from pyquil import get_qc
from pyquil.api._quantum_computer import _get_qvm_with_topology
from pyquil.gates import MEASURE, RESET
from pyquil.quil import Pragma, Program
from pyquil.paulis import sI, sX, sY, sZ
from pyquil.operator_estimation import ExperimentSetting, TensorProductState, Experiment, measure_observables, group_experiments
from pyquil.quilbase import Gate


class QPUDevice(QVMDevice):
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

        super(QVMDevice, self).__init__(num_wires, shots, **kwargs)

        if load_qc:
            self.qc = get_qc(device, as_qvm=False, connection=self.connection)
        else:
            self.qc = get_qc(device, as_qvm=True, connection=self.connection)

        self.active_reset = active_reset
        self.symmetrize_readout = symmetrize_readout
        self.calibrate_readout = calibrate_readout
        self.wiring = {i: q for i, q in enumerate(self.qc.qubits())}

    def expval(self, observable):
        wires = observable.wires
        # Single-qubit observable
        if len(wires) == 1:
            # identify Experiment Settings for each of the possible single-qubit observables
            wire = wires[0]
            qubit = self.wiring[wire]
            d_expt_settings = {
                "Identity": [ExperimentSetting(TensorProductState(), sI(qubit))],
                "PauliX": [ExperimentSetting(TensorProductState(), sX(qubit))],
                "PauliY": [ExperimentSetting(TensorProductState(), sY(qubit))],
                "PauliZ": [ExperimentSetting(TensorProductState(), sZ(qubit))],
                "Hadamard": [ExperimentSetting(TensorProductState(), float(np.sqrt(1/2)) * sX(qubit)),
                             ExperimentSetting(TensorProductState(), float(np.sqrt(1/2)) * sZ(qubit))]
            }
            # expectation values for single-qubit observables
            if observable.name in ["PauliX", "PauliY", "PauliZ", "Identity", "Hadamard"]:
                prep_prog = Program()
                for instr in self.program.instructions:
                    if isinstance(instr, Gate):
                        # split gate and wires -- assumes 1q and 2q gates
                        tup_gate_wires = instr.out().split(' ')
                        gate = tup_gate_wires[0]
                        str_instr = str(gate)
                        # map wires to qubits
                        for w in tup_gate_wires[1:]:
                            str_instr += f' {int(w)}'
                        prep_prog += Program(str_instr)

                if self.readout_error is not None:
                    prep_prog.define_noisy_readout(qubit, p00=self.readout_error[0],
                                                          p11=self.readout_error[1])

                # All observables are rotated and can be measured in the PauliZ basis
                tomo_expt = Experiment(settings=d_expt_settings["PauliZ"], program=prep_prog)
                grouped_tomo_expt = group_experiments(tomo_expt)
                meas_obs = list(measure_observables(self.qc, grouped_tomo_expt,
                                                    active_reset=self.active_reset,
                                                    symmetrize_readout=self.symmetrize_readout,
                                                    calibrate_readout=self.calibrate_readout))
                return np.sum([expt_result.expectation for expt_result in meas_obs])

            elif observable.name == 'Hermitian':
                # <H> = \sum_i w_i p_i
                Hkey = tuple(par[0].flatten().tolist())
                w = self._eigs[Hkey]['eigval']
                return w[0]*p0 + w[1]*p1

        return super().expval(observable)
