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
import warnings

import numpy as np
from pyquil import get_qc
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

from pennylane.operation import Tensor

from .qvm import QVMDevice


class QPUDevice(QVMDevice):
    r"""Forest QPU device for PennyLane.

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
        parametric_compilation (bool): a boolean value of whether or not to use parametric
            compilation.
    """
    name = "Forest QPU Device"
    short_name = "forest.qpu"
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hermitian"}
    pennylane_requires = ">=0.15.0"

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

        timeout = kwargs.pop("timeout", 10.0) # 10.0 is the pyquil default

        if shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")

        if load_qc:
            self.qc = get_qc(device, as_qvm=False, compiler_timeout=timeout)
        else:
            self.qc = get_qc(device, as_qvm=True, compiler_timeout=timeout)

        self.num_wires = len(self.qc.qubits())

        if wires is None:
            # infer the number of modes from the device specs
            # and use consecutive integer wire labels
            wires = range(self.num_wires)

        if isinstance(wires, int):
            raise ValueError(
                "Device has a fixed number of {} qubits. The wires argument can only be used "
                "to specify an iterable of wire labels.".format(self.num_wires)
            )

        if self.num_wires != len(wires):
            raise ValueError(
                "Device has a fixed number of {} qubits and "
                "cannot be created with {} wires.".format(self.num_wires, len(wires))
            )

        super(QVMDevice, self).__init__(wires, shots, **kwargs)

        self.active_reset = active_reset
        self.symmetrize_readout = symmetrize_readout
        self.calibrate_readout = calibrate_readout
        self.wiring = {i: q for i, q in enumerate(self.qc.qubits())}

    def expval(self, observable, shot_range, bin_size):
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
            elif (
                len(device_wires) > 1
                and isinstance(observable, Tensor)
                and not self.parametric_compilation
            ):

                # All observables are rotated to be measured in the Z-basis, so we just need to
                # check which wires exist in the observable, map them to physical qubits, and measure
                # the product of PauliZ operators on those qubits
                pauli_obs = sI()
                for label in device_wires.labels:
                    pauli_obs *= sZ(label)

            # Program preparing the state in which to measure observable
            prep_prog = Program()
            prep_prog += Program(prep_prog.reset())
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

            # TODO: This makes the QPU tests with parametric compilation disabled accurate
            # enough to pass, but probably isn't the right thing to do
            prep_prog.wrap_in_numshots_loop(self.shots)

            # Measure out multi-qubit observable
            tomo_expt = Experiment(
                settings=[ExperimentSetting(TensorProductState(), pauli_obs)], program=prep_prog, symmetrization=self.symmetrize_readout,
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

