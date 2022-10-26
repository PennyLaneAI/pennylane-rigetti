"""
Unit tests for the pyQVM simulator device.
"""
import logging

import numpy as np
import pennylane as qml
import pytest
from conftest import U2, BaseTest, H, I, U, Z, test_operation_map
from pennylane.circuit_graph import CircuitGraph
from pennylane.tape import QuantumTape

import pennylane_forest as plf

log = logging.getLogger(__name__)


# make tests deterministic
np.random.seed(42)


class TestPyQVMBasic(BaseTest):
    """Unit tests for the pyQVM simulator."""

    def test_identity_expectation(self, shots):
        """Test that identity expectation value (i.e. the trace) is 1"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-pyqvm", shots=shots)

        @qml.qnode(device=dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Identity(wires=[0])), qml.expval(qml.Identity(wires=[1]))

        res = circuit()

        # below are the analytic expectation values for this circuit (trace should always be 1)
        self.assertAllAlmostEqual(res, np.array([1, 1]), delta=3 / np.sqrt(shots))

    def test_pauliz_expectation(self, shots):
        """Test that PauliZ expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-pyqvm", shots=shots)

        @qml.qnode(device=dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=[0])), qml.expval(qml.PauliZ(wires=[1]))

        res = circuit()
        # below are the analytic expectation values for this circuit
        self.assertAllAlmostEqual(
            res, np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]), delta=3 / np.sqrt(shots)
        )

    def test_paulix_expectation(self, shots):
        """Test that PauliX expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-pyqvm", shots=shots)

        @qml.qnode(device=dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=[0])), qml.expval(qml.PauliX(wires=[1]))

        res = circuit()
        # below are the analytic expectation values for this circuit
        self.assertAllAlmostEqual(
            res, np.array([np.sin(theta) * np.sin(phi), np.sin(phi)]), delta=3 / np.sqrt(shots)
        )

    def test_pauliy_expectation(self, shots):
        """Test that PauliY expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-pyqvm", shots=shots)

        @qml.qnode(device=dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(wires=[0])), qml.expval(qml.PauliY(wires=[1]))

        res = circuit()

        # below are the analytic expectation values for this circuit
        self.assertAllAlmostEqual(
            res, np.array([0, -np.cos(theta) * np.sin(phi)]), delta=3 / np.sqrt(shots)
        )

    def test_hadamard_expectation(self, shots):
        """Test that Hadamard expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-pyqvm", shots=shots)

        @qml.qnode(device=dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hadamard(wires=[0])), qml.expval(qml.Hadamard(wires=[1]))

        res = circuit()

        # below are the analytic expectation values for this circuit
        expected = np.array(
            [np.sin(theta) * np.sin(phi) + np.cos(theta), np.cos(theta) * np.cos(phi) + np.sin(phi)]
        ) / np.sqrt(2)
        self.assertAllAlmostEqual(res, expected, delta=3 / np.sqrt(shots))

    def test_hermitian_expectation(self, shots):
        """Test that arbitrary Hermitian expectation values are correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-pyqvm", shots=5 * shots)

        @qml.qnode(device=dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(H, wires=[0])), qml.expval(qml.Hermitian(H, wires=[1]))

        res = circuit()
        # below are the analytic expectation values for this circuit with arbitrary
        # Hermitian observable H
        a = H[0, 0]
        re_b = H[0, 1].real
        d = H[1, 1]
        ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
        expected = np.array([ev1, ev2])

        self.assertAllAlmostEqual(res, expected, delta=3 / np.sqrt(shots))

    def test_multi_qubit_hermitian_expectation(self, shots):
        """Test that arbitrary multi-qubit Hermitian expectation values are correct"""
        theta = np.random.random()
        phi = np.random.random()

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        dev = plf.QVMDevice(device="2q-pyqvm", shots=10 * shots)

        @qml.qnode(device=dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(A, wires=[0, 1]))

        res = circuit()
        # below is the analytic expectation value for this circuit with arbitrary
        # Hermitian observable A
        expected = 0.5 * (
            6 * np.cos(theta) * np.sin(phi)
            - np.sin(theta) * (8 * np.sin(phi) + 7 * np.cos(phi) + 3)
            - 2 * np.sin(phi)
            - 6 * np.cos(phi)
            - 6
        )

        self.assertAllAlmostEqual(res, expected, delta=6 / np.sqrt(shots))

    def test_var(self, shots):
        """Tests for variance calculation"""
        dev = plf.QVMDevice(device="2q-pyqvm", shots=shots)

        phi = 0.543
        theta = 0.6543

        @qml.qnode(device=dev)
        def circuit():
            qml.RX(phi, wires=[0])
            qml.RY(theta, wires=[0])
            return qml.var(qml.PauliZ(wires=[0]))

        var = circuit()
        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))

        self.assertAlmostEqual(var, expected, delta=3 / np.sqrt(shots))

    def test_var_hermitian(self, shots):
        """Tests for variance calculation using an arbitrary Hermitian observable"""
        dev = plf.QVMDevice(device="2q-pyqvm", shots=100 * shots)

        phi = 0.543
        theta = 0.6543

        # test correct variance for <A> of a rotated state
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])

        @qml.qnode(device=dev)
        def circuit():
            qml.RX(phi, wires=[0])
            qml.RY(theta, wires=[0])
            return qml.var(qml.Hermitian(A, wires=[0]))

        var = circuit()
        expected = 0.5 * (
            2 * np.sin(2 * theta) * np.cos(phi) ** 2
            + 24 * np.sin(phi) * np.cos(phi) * (np.sin(theta) - np.cos(theta))
            + 35 * np.cos(2 * phi)
            + 39
        )

        self.assertAlmostEqual(var, expected, delta=0.3)

    @pytest.mark.parametrize("gate", plf.QVMDevice._operation_map)
    def test_apply(self, gate, apply_unitary, shots):
        """Test the application of gates"""
        dev = plf.QVMDevice(device="3q-pyqvm", shots=shots)

        try:
            # get the equivalent pennylane operation class
            op = getattr(qml.ops, gate)
        except AttributeError:
            # get the equivalent pennylane-forest operation class
            op = getattr(plf, gate)

        # the list of wires to apply the operation to
        w = list(range(op.num_wires))

        if gate == "QubitUnitary":
            p = [np.array(U)]
            w = [0]
            state = apply_unitary(U, 3)
        elif gate == "BasisState":
            p = [np.array([1, 1, 1])]
            state = np.array([0, 0, 0, 0, 0, 0, 0, 1])
            w = list(range(dev.num_wires))
        else:
            p = [0.432423, 2, 0.324][: op.num_params]
            fn = test_operation_map[gate]
            # if the default.qubit is an operation accepting parameters,
            # initialise it using the parameters generated above.
            # otherwise, the operation is simply an array.
            O = fn(*p) if callable(fn) else fn
            # calculate the expected output
            state = apply_unitary(O, 3)

        @qml.qnode(device=dev)
        def circuit():
            if p:
                op(*p, wires=w)  # parametrized gate
            else:
                op(wires=w)  # non-parametrized gate
            return qml.expval(qml.PauliZ(0))

        res = circuit()
        expected = np.vdot(state, np.kron(np.kron(Z, I), I) @ state)

        # verify the device is now in the expected state
        # Note we have increased the tolerance here, since we are only
        # performing 1024 shots.
        self.assertAllAlmostEqual(res, expected, delta=3 / np.sqrt(shots))


class TestQVMIntegration(BaseTest):
    """Test the pyQVM simulator works correctly from the PennyLane frontend."""

    # pylint: disable=no-self-use

    def test_qubit_unitary(self, shots):
        """Test that an arbitrary unitary operation works"""
        dev1 = qml.device("forest.qvm", device="3q-pyqvm", shots=shots)
        dev2 = qml.device("forest.qvm", device="9q-square-pyqvm", shots=shots)

        def circuit():
            """Reference QNode"""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.QubitUnitary(U2, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit1 = qml.QNode(circuit, dev1)
        circuit2 = qml.QNode(circuit, dev2)

        out_state = U2 @ np.array([1, 0, 0, 1]) / np.sqrt(2)
        obs = np.kron(np.array([[1, 0], [0, -1]]), I)

        self.assertAllAlmostEqual(
            circuit1(), np.vdot(out_state, obs @ out_state), delta=3 / np.sqrt(shots)
        )
        self.assertAllAlmostEqual(
            circuit2(), np.vdot(out_state, obs @ out_state), delta=3 / np.sqrt(shots)
        )

    @pytest.mark.parametrize("device", ["2q-pyqvm"])
    def test_one_qubit_wavefunction_circuit(self, device, shots):
        """Test that the wavefunction plugin provides correct result for simple circuit."""
        shots = 10000
        dev = qml.device("forest.qvm", device=device, shots=shots)

        a = 0.543
        b = 0.123
        c = 0.987

        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.BasisState(np.array([1]), wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        self.assertAlmostEqual(circuit(a, b, c), np.cos(a) * np.sin(b), delta=5 / np.sqrt(shots))
