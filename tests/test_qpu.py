"""
Unit tests for the QPU device.
"""
import logging

import pytest

import pennylane as qml

from conftest import BaseTest


log = logging.getLogger(__name__)


class TestQPUIntegration(BaseTest):
    """Test the wavefunction simulator works correctly from the PennyLane frontend."""
    #pylint: disable=no-self-use

    def test_load_qpu_device(self):
        """Test that the QPU device loads correctly"""
        dev = qml.device('forest.qpu', device='Aspen-1-2Q-B', load_qc=False)
        self.assertEqual(dev.num_wires, 2)
        self.assertEqual(dev.shots, 1024)
        self.assertEqual(dev.short_name, 'forest.qpu')

    def test_load_virtual_qpu_device(self):
        """Test that the QPU simulators load correctly"""
        qml.device('forest.qpu', device='Aspen-1-2Q-B', load_qc=False)

    def test_qpu_args(self):
        """Test that the QPU plugin requires correct arguments"""
        with pytest.raises(ValueError, match="QPU device does not support a wires parameter."):
            qml.device('forest.qpu', device='Aspen-1-7Q-B', wires=2)

        with pytest.raises(TypeError, match="missing 1 required positional arguments: 'device'"):
            qml.device('forest.qpu')

        with pytest.raises(ValueError, match="Number of shots must be a postive integer"):
            qml.device('forest.qpu', 'Aspen-1-7Q-B', shots=0)
