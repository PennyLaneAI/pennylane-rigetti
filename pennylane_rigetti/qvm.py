"""
QVM Device
==========

**Module name:** :mod:`pennylane_rigetti.qvm`

.. currentmodule:: pennylane_rigetti.qvm

This module contains the :class:`~.QVMDevice` class, a PennyLane device that allows
evaluation and differentiation of Rigetti's Quantum Virtual Machines (QVMs)
using PennyLane.

Classes
-------

.. autosummary::
   QVMDevice

Code details
~~~~~~~~~~~~
"""

import networkx as nx
from pyquil import get_qc
from pyquil.api import QuantumComputer, QuantumExecutable
from pyquil.api._quantum_computer import _get_qvm_with_topology
from qcs_api_client.client import QCSClientConfiguration

from .qc import QuantumComputerDevice


class QVMDevice(QuantumComputerDevice):
    r"""Rigetti QVM device for PennyLane.

    This device supports both the Rigetti Lisp QVM, as well as the built-in pyQuil pyQVM.
    If using the pyQVM, the ``qvm_url`` QVM server url keyword argument does not need to
    be set.

    Args:
        device (Union[str, nx.Graph]): the name or topology of the device to initialise.

            * ``Nq-qvm``: for a fully connected/unrestricted N-qubit QVM
            * ``9q-square-qvm``: a :math:`9\times 9` lattice.
            * ``Nq-pyqvm`` or ``9q-square-pyqvm``, for the same as the above but run
              via the built-in pyQuil pyQVM device.
            * Any other supported Rigetti device architecture.
            * Graph topology representing the device architecture.

        shots (None, int, list[int]): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. If ``None``, the device calculates probability, expectation values,
            and variances analytically. If an integer, it specifies the number of samples to estimate these quantities.
            If a list of integers is passed, the circuit evaluations are batched over the list of shots.
        wires (Iterable[Number, str]): Iterable that contains unique labels for the
            qubits as numbers or strings (i.e., ``['q1', ..., 'qN']``).
            The number of labels must match the number of qubits accessible on the backend.
            If not provided, qubits are addressed as consecutive integers [0, 1, ...], and their number
            is inferred from the backend.
        noisy (bool): set to ``True`` to add noise models to your QVM.

    Keyword args:
        compiler_timeout (int): number of seconds to wait for a response from quilc (default 10).
        execution_timeout (int): number of seconds to wait for a response from the QVM (default 10).
        parametric_compilation (bool): a boolean value of whether or not to use parametric
            compilation.
    """

    name = "Rigetti QVM Device"
    short_name = "rigetti.qvm"

    def __init__(self, device, *, shots=1000, wires=None, noisy=False, **kwargs):
        if shots is None:
            raise ValueError("QVM device cannot be used for analytic computations.")

        self.noisy = noisy

        super().__init__(
            device, wires=wires, shots=shots, noisy=noisy, active_reset=False, **kwargs
        )

    def get_qc(self, device, **kwargs) -> QuantumComputer:
        if isinstance(device, nx.Graph):
            client_configuration = QCSClientConfiguration.load()
            return _get_qvm_with_topology(
                name="device",
                topology=device,
                noisy=self.noisy,
                client_configuration=client_configuration,
                qvm_type="qvm",
                compiler_timeout=kwargs.get("compiler_timeout", 10.0),  # 10.0 is the pyQuil default
                execution_timeout=kwargs.get(
                    "execution_timeout", 10.0
                ),  # 10.0 is the pyQuil default
            )
        return get_qc(device, as_qvm=True, noisy=self.noisy, **kwargs)

    def compile(self) -> QuantumExecutable:
        """Skips compilation for pyqvm devices as it isn't required."""
        if "pyqvm" in self.qc.name:
            return self.prog
        return super().compile()
