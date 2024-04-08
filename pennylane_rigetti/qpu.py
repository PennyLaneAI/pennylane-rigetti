"""
QPU Device
==========

**Module name:** :mod:`pennylane_rigetti.qpu`

.. currentmodule:: pennylane_rigetti.qpu

This module contains the :class:`~.QPUDevice` class, a PennyLane device that allows
evaluation and differentiation of Rigetti's Quantum Processing Units (QPUs)
using PennyLane.

Classes
-------

.. autosummary::
   QPUDevice

Code details
~~~~~~~~~~~~
"""

import warnings

import numpy as np
from pennylane.measurements import Expectation
from pennylane.operation import Tensor
from pennylane.ops import Prod
from pennylane.tape import QuantumTape
from pyquil import get_qc
from pyquil.api import QuantumComputer
from pyquil.experiment import SymmetrizationLevel
from pyquil.operator_estimation import (
    Experiment,
    ExperimentSetting,
    TensorProductState,
    group_experiments,
    measure_observables,
)
from pyquil.paulis import sI, sZ
from pyquil.quil import Program
from pyquil.quilbase import Gate

from .qc import QuantumComputerDevice


class QPUDevice(QuantumComputerDevice):
    r"""Rigetti QPU device for PennyLane.

    Args:
        device (str): the name of the device to initialise.
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of observables.
         wires (Iterable[Number, str]): Iterable that contains unique labels for the
            qubits as numbers or strings (i.e., ``['q1', ..., 'qN']``).
            The number of labels must match the number of qubits accessible on the backend.
            If not provided, qubits are addressed as consecutive integers ``[0, 1, ...]``, and their number
            is inferred from the backend.
        active_reset (bool): whether to actively reset qubits instead of waiting for
            for qubits to decay to the ground state naturally.
            Setting this to ``True`` results in a significantly faster expectation value
            evaluation when the number of shots is larger than ~1000.
        load_qc (bool): set to False to avoid getting the quantum computing
            device on initialization. This is convenient if not currently connected to the QPU.
        readout_error (list): specifies the conditional probabilities [p(0|0), p(1|1)], where
            p(i|j) denotes the prob of reading out i having sampled j; can be set to `None` if no
            readout errors need to be simulated; can only be set for the QPU-as-a-QVM
        symmetrize_readout (pyquil.experiment.SymmetrizationLevel): method to perform readout symmetrization, using exhaustive
            symmetrization by default
        calibrate_readout (str): method to perform calibration for readout error mitigation,
            normalizing by the expectation value in the +1-eigenstate of the observable by default

    Keyword args:
        compiler_timeout (int): number of seconds to wait for a response from quilc (default 10).
        execution_timeout (int): number of seconds to wait for a response from the QVM (default 10).
        parametric_compilation (bool): a boolean value of whether or not to use parametric
            compilation.
    """

    name = "Rigetti QPU Device"
    short_name = "rigetti.qpu"

    def __init__(
        self,
        device,
        *,
        shots=1000,
        wires=None,
        active_reset=True,
        load_qc=True,
        readout_error=None,
        symmetrize_readout=SymmetrizationLevel.EXHAUSTIVE,
        calibrate_readout="plus-eig",
        **kwargs,
    ):
        if readout_error is not None and load_qc:
            raise ValueError("Readout error cannot be set on the physical QPU")

        self.readout_error = readout_error

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

        self.as_qvm = not load_qc
        self.symmetrize_readout = symmetrize_readout
        self.calibrate_readout = calibrate_readout
        self._skip_generate_samples = False

        super().__init__(device, wires=wires, shots=shots, active_reset=active_reset, **kwargs)

    def get_qc(self, device, **kwargs) -> QuantumComputer:
        return get_qc(device, as_qvm=self.as_qvm, **kwargs)

    def expval(self, observable, shot_range=None, bin_size=None):
        # translate operator wires to wire labels on the device
        device_wires = self.map_wires(observable.wires)

        # `measure_observables` called only when parametric compilation is turned off
        if not self.parametric_compilation:
            # Single-qubit observable
            if len(device_wires) == 1:
                # Ensure sensible observable
                assert observable.name in [
                    "PauliX",
                    "PauliY",
                    "PauliZ",
                    "Identity",
                    "Hadamard",
                ], "Unknown observable"

                # Create appropriate PauliZ operator
                wire = device_wires.labels[0]
                pauli_obs = sZ(wire)

            # Multi-qubit observable
            elif len(device_wires) > 1 and isinstance(observable, (Tensor, Prod)):
                # All observables are rotated to be measured in the Z-basis, so we just need to
                # check which wires exist in the observable, map them to physical qubits, and measure
                # the product of PauliZ operators on those qubits
                pauli_obs = sI()
                for label in device_wires.labels:
                    pauli_obs *= sZ(label)

            # Program preparing the state in which to measure observable
            prep_prog = Program("RESET 0")
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
                for label in device_wires.labels:
                    prep_prog.define_noisy_readout(
                        label, p00=self.readout_error[0], p11=self.readout_error[1]
                    )

            prep_prog.wrap_in_numshots_loop(self.shots)

            # Measure out multi-qubit observable
            tomo_expt = Experiment(
                settings=[ExperimentSetting(TensorProductState(), pauli_obs)],
                program=prep_prog,
                symmetrization=self.symmetrize_readout,
            )
            grouped_tomo_expt = group_experiments(tomo_expt)
            meas_obs = list(
                measure_observables(
                    self.qc,
                    grouped_tomo_expt,
                    calibrate_readout=self.calibrate_readout,
                )
            )

            # Return the estimated expectation value
            return np.sum([expt_result.expectation for expt_result in meas_obs])

        # Calculation of expectation value without using `measure_observables`
        return super().expval(observable, shot_range, bin_size)

    def execute(self, circuit: QuantumTape, **kwargs):
        self._skip_generate_samples = (
            all(mp.return_type is Expectation for mp in circuit.measurements)
            and not self.parametric_compilation
        )

        return super().execute(circuit, **kwargs)

    def generate_samples(self):
        return None if self._skip_generate_samples else super().generate_samples()
