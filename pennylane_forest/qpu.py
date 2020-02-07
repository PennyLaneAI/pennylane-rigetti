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
import warnings

import numpy as np
from pyquil import get_qc
from pyquil.operator_estimation import (
    Experiment,
    ExperimentSetting,
    TensorProductState,
    group_experiments,
    measure_observables,
)
from pyquil.paulis import sI, sX, sY, sZ
from pyquil.quil import Program
from pyquil.quilbase import Gate

from ._version import __version__
from .qvm import QVMDevice


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
        calibrate_readout (str): method to perform calibration for readout error mitigation,
            normalizing by the expectation value in the +1-eigenstate of the observable by default

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
        timeout (int): number of seconds to wait for a response from the client.
        parametric_compilation (bool): a boolean value of whether or not to use parametric
            compilation.
    """
    name = "Forest QPU Device"
    short_name = "forest.qpu"
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hermitian"}

    def __init__(
        self,
        device,
        *,
        shots=1024,
        active_reset=True,
        load_qc=True,
        readout_error=None,
        symmetrize_readout="exhaustive",
        calibrate_readout="plus-eig",
        **kwargs,
    ):

        if readout_error is not None and load_qc:
            raise ValueError("Readout error cannot be set on the physical QPU")

        self.readout_error = readout_error

        self._eigs = {}

        self._compiled_program = None
        """Union[None, pyquil.ExecutableDesignator]: the latest compiled program. If parametric
        compilation is turned on, this will be a parametric program."""

        if kwargs.get("parametric_compilation", False):
            # Raise a warning if parametric compilation was explicitly turned on by the user
            # about turning the operator estimation off

            # TODO: Remove the warning and toggling once a migration to the new operator estimation
            # API has been executed. This new API provides compatibility between parametric
            # compilation and operator estimation.
            warnings.warn(
                "Parametric compilation is currently not supported with operator"
                "estimation. Operator estimation is being turned off."
            )

        self.parametric_compilation = kwargs.get("parametric_compilation", True)

        if self.parametric_compilation:
            self._compiled_program_dict = {}
            """dict[int, pyquil.ExecutableDesignator]: stores circuit hashes associated
                with the corresponding compiled programs."""

            self._parameter_map = {}
            """dict[str, float]: stores the string of symbolic parameters associated with
                their numeric values. This map will be used to bind parameters in a parametric
                program using PyQuil."""

            self._parameter_reference_map = {}
            """dict[str, pyquil.quilatom.MemoryReference]: stores the string of symbolic
                parameters associated with their PyQuil memory references."""

        timeout = kwargs.pop("timeout", None)

        if "wires" in kwargs:
            raise ValueError("QPU device does not support a wires parameter.")

        if shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")

        aspen_match = re.match(r"Aspen-\d+-([\d]+)Q", device)
        num_wires = int(aspen_match.groups()[0])

        super(QVMDevice, self).__init__(num_wires, shots, **kwargs)

        if load_qc:
            self.qc = get_qc(device, as_qvm=False, connection=self.connection)
            if timeout is not None:
                self.qc.compiler.quilc_client.timeout = timeout
        else:
            self.qc = get_qc(device, as_qvm=True, connection=self.connection)
            if timeout is not None:
                self.qc.compiler.client.timeout = timeout

        self.active_reset = active_reset
        self.symmetrize_readout = symmetrize_readout
        self.calibrate_readout = calibrate_readout
        self.wiring = {i: q for i, q in enumerate(self.qc.qubits())}

    def expval(self, observable):
        wires = observable.wires

        if len(wires) == 1 and not self.parametric_compilation:
            # Single-qubit observable when parametric compilation is turned off

            # identify Experiment Settings for each of the possible single-qubit observables
            wire = wires[0]
            qubit = self.wiring[wire]
            d_expt_settings = {
                "Identity": [ExperimentSetting(TensorProductState(), sI(qubit))],
                "PauliX": [ExperimentSetting(TensorProductState(), sX(qubit))],
                "PauliY": [ExperimentSetting(TensorProductState(), sY(qubit))],
                "PauliZ": [ExperimentSetting(TensorProductState(), sZ(qubit))],
                "Hadamard": [
                    ExperimentSetting(TensorProductState(), float(np.sqrt(1 / 2)) * sX(qubit)),
                    ExperimentSetting(TensorProductState(), float(np.sqrt(1 / 2)) * sZ(qubit)),
                ],
            }

            if observable.name in ["PauliX", "PauliY", "PauliZ", "Identity", "Hadamard"]:
                # expectation values for single-qubit observables

                prep_prog = Program()
                for instr in self.program.instructions:
                    if isinstance(instr, Gate):
                        # split gate and wires -- assumes 1q and 2q gates
                        tup_gate_wires = instr.out().split(" ")
                        gate = tup_gate_wires[0]
                        str_instr = str(gate)
                        # map wires to qubits
                        for w in tup_gate_wires[1:]:
                            str_instr += f" {int(w)}"
                        prep_prog += Program(str_instr)

                if self.readout_error is not None:
                    prep_prog.define_noisy_readout(
                        qubit, p00=self.readout_error[0], p11=self.readout_error[1]
                    )

                # All observables are rotated and can be measured in the PauliZ basis
                tomo_expt = Experiment(settings=d_expt_settings["PauliZ"], program=prep_prog)
                grouped_tomo_expt = group_experiments(tomo_expt)
                meas_obs = list(
                    measure_observables(
                        self.qc,
                        grouped_tomo_expt,
                        active_reset=self.active_reset,
                        symmetrize_readout=self.symmetrize_readout,
                        calibrate_readout=self.calibrate_readout,
                    )
                )
                return np.sum([expt_result.expectation for expt_result in meas_obs])

            elif observable.name == "Hermitian":
                # <H> = \sum_i w_i p_i
                Hkey = tuple(par[0].flatten().tolist())
                w = self._eigs[Hkey]["eigval"]
                return w[0] * p0 + w[1] * p1
        return super().expval(observable)
