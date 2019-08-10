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

    def __init__(self, device, *, shots=1024, active_reset=False, load_qc=True, wires=wires, **kwargs):
        self._eigs = {}

        if "wires" in kwargs:
            raise ValueError("QPU device does not support a wires parameter.")

        if shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")

        aspen_match = re.match(r"Aspen-\d+-([\d]+)Q", device)
        # num_wires = int(aspen_match.groups()[0])
        num_wires = wires

        super(QVMDevice, self).__init__(
            num_wires, shots, **kwargs
        )  # pylint: disable=bad-super-call

        if load_qc:
            self.qc = get_qc(device, as_qvm=False, connection=self.connection)

        self.active_reset = active_reset
