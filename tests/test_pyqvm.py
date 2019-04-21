"""
Unit tests for the pyQVM simulator device.
"""
import logging

import networkx as nx
import pytest

import pennylane as qml
from pennylane import numpy as np

from conftest import BaseTest
from conftest import I, Z, H, U, U2, test_operation_map

import pennylane_forest as plf


log = logging.getLogger(__name__)


class TestPyQVMBasic(BaseTest):
    """Unit tests for the pyQVM simulator."""
    # pylint: disable=protected-access

    def test_identity_expectation(self, shots):
        """Test that identity expectation value (i.e. the trace) is 1"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device='2q-pyqvm', shots=shots)
        dev.apply('RX', wires=[0], par=[theta])
        dev.apply('RX', wires=[1], par=[phi])
        dev.apply('CNOT', wires=[0, 1], par=[])

        O = qml.expval.qubit.Identity
        name = 'Identity'

        dev._expval_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        res = dev.pre_expval()

        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])

        # below are the analytic expectation values for this circuit (trace should always be 1)
        self.assertAllAlmostEqual(res, np.array([1, 1]), delta=3/np.sqrt(shots))

    def test_pauliz_expectation(self, shots):
        """Test that PauliZ expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device='2q-pyqvm', shots=shots)
        dev.apply('RX', wires=[0], par=[theta])
        dev.apply('RX', wires=[1], par=[phi])
        dev.apply('CNOT', wires=[0, 1], par=[])

        O = qml.expval.PauliZ
        name = 'PauliZ'

        dev._expval_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        res = dev.pre_expval()

        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])

        # below are the analytic expectation values for this circuit
        self.assertAllAlmostEqual(res, np.array([np.cos(theta), np.cos(theta)*np.cos(phi)]), delta=3/np.sqrt(shots))

    def test_paulix_expectation(self, shots):
        """Test that PauliX expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device='2q-pyqvm', shots=shots)
        dev.apply('RY', wires=[0], par=[theta])
        dev.apply('RY', wires=[1], par=[phi])
        dev.apply('CNOT', wires=[0, 1], par=[])

        O = qml.expval.PauliX
        name = 'PauliX'

        dev._expval_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        dev.pre_expval()

        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])
        # below are the analytic expectation values for this circuit
        self.assertAllAlmostEqual(res, np.array([np.sin(theta)*np.sin(phi), np.sin(phi)]), delta=3/np.sqrt(shots))

    def test_pauliy_expectation(self, shots):
        """Test that PauliY expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device='2q-pyqvm', shots=shots)
        dev.apply('RX', wires=[0], par=[theta])
        dev.apply('RX', wires=[1], par=[phi])
        dev.apply('CNOT', wires=[0, 1], par=[])

        O = qml.expval.PauliY
        name = 'PauliY'

        dev._expval_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        dev.pre_expval()

        # below are the analytic expectation values for this circuit
        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])
        self.assertAllAlmostEqual(res, np.array([0, -np.cos(theta)*np.sin(phi)]), delta=3/np.sqrt(shots))

    def test_hadamard_expectation(self, shots):
        """Test that Hadamard expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device='2q-pyqvm', shots=shots)
        dev.apply('RY', wires=[0], par=[theta])
        dev.apply('RY', wires=[1], par=[phi])
        dev.apply('CNOT', wires=[0, 1], par=[])

        O = qml.expval.Hadamard
        name = 'Hadamard'

        dev._expval_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        dev.pre_expval()

        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])
        # below are the analytic expectation values for this circuit
        expected = np.array([np.sin(theta)*np.sin(phi)+np.cos(theta), np.cos(theta)*np.cos(phi)+np.sin(phi)])/np.sqrt(2)
        self.assertAllAlmostEqual(res, expected, delta=3/np.sqrt(shots))

    def test_hermitian_expectation(self, shots):
        """Test that arbitrary Hermitian expectation values are correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device='2q-pyqvm', shots=shots)
        dev.apply('RY', wires=[0], par=[theta])
        dev.apply('RY', wires=[1], par=[phi])
        dev.apply('CNOT', wires=[0, 1], par=[])

        O = qml.expval.qubit.Hermitian
        name = 'Hermitian'

        dev._expval_queue = [O(H, wires=[0], do_queue=False), O(H, wires=[1], do_queue=False)]
        dev.pre_expval()

        res = np.array([dev.expval(name, [0], [H]), dev.expval(name, [1], [H])])

        # below are the analytic expectation values for this circuit with arbitrary
        # Hermitian observable H
        a = H[0, 0]
        re_b = H[0, 1].real
        d = H[1, 1]
        ev1 = ((a-d)*np.cos(theta)+2*re_b*np.sin(theta)*np.sin(phi)+a+d)/2
        ev2 = ((a-d)*np.cos(theta)*np.cos(phi)+2*re_b*np.sin(phi)+a+d)/2
        expected = np.array([ev1, ev2])

        self.assertAllAlmostEqual(res, expected, delta=3/np.sqrt(shots))

    @pytest.mark.parametrize("gate", plf.QVMDevice._operation_map) #pylint: disable=protected-access
    def test_apply(self, gate, apply_unitary, shots):
        """Test the application of gates"""
        dev = plf.QVMDevice(device='3q-pyqvm', shots=shots)

        try:
            # get the equivalent pennylane operation class
            op = getattr(qml.ops, gate)
        except AttributeError:
            # get the equivalent pennylane-forest operation class
            op = getattr(plf, gate)

        # the list of wires to apply the operation to
        w = list(range(op.num_wires))

        if op.par_domain == 'A':
            # the parameter is an array
            if gate == 'QubitUnitary':
                p = [U]
                w = [0]
                state = apply_unitary(U, 3)
            elif gate == 'BasisState':
                p = [np.array([1, 1, 1])]
                state = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        else:
            p = [0.432423, 2, 0.324][:op.num_params]
            fn = test_operation_map[gate]

            if callable(fn):
                # if the default.qubit is an operation accepting parameters,
                # initialise it using the parameters generated above.
                O = fn(*p)
            else:
                # otherwise, the operation is simply an array.
                O = fn

            # calculate the expected output
            state = apply_unitary(O, 3)

        dev.apply(gate, wires=w, par=p)
        dev._expval_queue = []
        dev.pre_expval()

        res = dev.expval('PauliZ', wires=[0], par=None)
        expected = np.vdot(state, np.kron(np.kron(Z, I), I) @ state)

        # verify the device is now in the expected state
        # Note we have increased the tolerance here, since we are only
        # performing 1024 shots.
        self.assertAllAlmostEqual(res, expected, delta=3/np.sqrt(shots))


class TestQVMIntegration(BaseTest):
    """Test the pyQVM simulator works correctly from the PennyLane frontend."""
    #pylint: disable=no-self-use

    def test_qubit_unitary(self, shots):
        """Test that an arbitrary unitary operation works"""
        dev1 = qml.device('forest.qvm', device='3q-pyqvm', shots=shots)
        dev2 = qml.device('forest.qvm', device='9q-square-pyqvm', shots=shots)

        def circuit():
            """Reference QNode"""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.QubitUnitary(U2, wires=[0, 1])
            return qml.expval.PauliZ(0)

        circuit1 = qml.QNode(circuit, dev1)
        circuit2 = qml.QNode(circuit, dev2)

        out_state = U2 @ np.array([1, 0, 0, 1])/np.sqrt(2)
        obs = np.kron(np.array([[1, 0], [0, -1]]), I)

        self.assertAllAlmostEqual(circuit1(), np.vdot(out_state, obs @ out_state), delta=3/np.sqrt(shots))
        self.assertAllAlmostEqual(circuit2(), np.vdot(out_state, obs @ out_state), delta=3/np.sqrt(shots))

    @pytest.mark.parametrize('device', ['2q-pyqvm'])
    def test_one_qubit_wavefunction_circuit(self, device):
        """Test that the wavefunction plugin provides correct result for simple circuit"""
        shots = 100000
        dev = qml.device('forest.qvm', device=device, shots=shots)

        a = 0.543
        b = 0.123
        c = 0.987

        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.BasisState(np.array([1]), wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval.PauliZ(0)

        self.assertAlmostEqual(circuit(a, b, c), np.cos(a)*np.sin(b), delta=3/np.sqrt(shots))
