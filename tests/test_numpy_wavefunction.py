"""
Unit tests for the wavefunction simulator device.
"""
import logging

import pytest

import pennylane as qml
from pennylane import numpy as np

from conftest import BaseTest
from conftest import I, U, U2, SWAP, CNOT, U_toffoli, H, test_operation_map

import pennylane_forest as plf


log = logging.getLogger(__name__)


class TestWavefunctionBasic(BaseTest):
    """Unit tests for the NumPy wavefunction simulator."""

    @pytest.mark.parametrize("ev", plf.NumpyWavefunctionDevice.observables)
    def test_ev(self, ev, tol):
        """Test that expectation values are calculated correctly"""
        dev = plf.NumpyWavefunctionDevice(wires=2)

        # start in the following initial state
        dev.state = np.array([1, 0, 1, 1]) / np.sqrt(3)
        dev.active_wires = {0, 1}

        # get the equivalent pennylane operation class
        op = getattr(qml.ops, ev)

        O = test_operation_map[ev]

        # calculate the expected output
        if op.num_wires == 1 or op.num_wires == 0:
            expected_out = dev.state.conj() @ np.kron(O, I) @ dev.state
        elif op.num_wires == 2:
            expected_out = dev.state.conj() @ O @ dev.state

        res = dev.ev(O, wires=[0])

        # verify the device is now in the expected state
        self.assertAllAlmostEqual(res, expected_out, delta=tol)

    def test_var(self, tol):
        """Tests for variance calculation"""
        dev = plf.NumpyWavefunctionDevice(wires=2)
        dev.active_wires = {0}

        phi = 0.543
        theta = 0.6543

        # test correct variance for <Z> of a rotated state
        dev.apply("RX", wires=[0], par=[phi])
        dev.apply("RY", wires=[0], par=[theta])
        dev.pre_measure()

        var = dev.var("PauliZ", [0], [])
        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))

        self.assertAlmostEqual(var, expected, delta=tol)

    def test_var_hermitian(self, tol):
        """Tests for variance calculation using an arbitrary Hermitian observable"""
        dev = plf.NumpyWavefunctionDevice(wires=2)
        dev.active_wires = {0}

        phi = 0.543
        theta = 0.6543

        # test correct variance for <H> of a rotated state
        H = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        dev.apply("RX", wires=[0], par=[phi])
        dev.apply("RY", wires=[0], par=[theta])
        dev.pre_measure()

        var = dev.var("Hermitian", [0], [H])
        expected = 0.5 * (
            2 * np.sin(2 * theta) * np.cos(phi) ** 2
            + 24 * np.sin(phi) * np.cos(phi) * (np.sin(theta) - np.cos(theta))
            + 35 * np.cos(2 * phi)
            + 39
        )

        self.assertAlmostEqual(var, expected, delta=tol)

    @pytest.mark.parametrize(
        "gate", plf.NumpyWavefunctionDevice._operation_map
    )  # pylint: disable=protected-access
    def test_apply(self, gate, apply_unitary, tol):
        """Test the application of gates to a state"""
        dev = plf.NumpyWavefunctionDevice(wires=3)

        try:
            # get the equivalent pennylane operation class
            op = getattr(qml.ops, gate)
        except AttributeError:
            # get the equivalent pennylane-forest operation class
            op = getattr(plf, gate)

        # the list of wires to apply the operation to
        w = list(range(op.num_wires))

        if op.par_domain == "A":
            # the parameter is an array
            if gate == "QubitUnitary":
                p = [U]
                w = [0]
                expected_out = apply_unitary(U, 3)
            elif gate == "BasisState":
                p = [np.array([1, 1, 1])]
                expected_out = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        else:
            p = [0.432423, 2, 0.324][: op.num_params]
            fn = test_operation_map[gate]

            if callable(fn):
                # if the default.qubit is an operation accepting parameters,
                # initialise it using the parameters generated above.
                O = fn(*p)
            else:
                # otherwise, the operation is simply an array.
                O = fn

            # calculate the expected output
            expected_out = apply_unitary(O, 3)

        dev.apply(gate, wires=w, par=p)
        dev.pre_measure()

        # verify the device is now in the expected state
        self.assertAllAlmostEqual(dev.state, expected_out, delta=tol)


class TestWavefunctionIntegration(BaseTest):
    """Test the NumPy wavefunction simulator works correctly from the PennyLane frontend."""

    # pylint:disable=no-self-use

    def test_load_wavefunction_device(self):
        """Test that the wavefunction device loads correctly"""
        dev = qml.device("forest.numpy_wavefunction", wires=2)
        self.assertEqual(dev.num_wires, 2)
        self.assertEqual(dev.shots, 0)
        self.assertEqual(dev.short_name, "forest.numpy_wavefunction")

    def test_program_property(self):
        """Test that the program property works as expected"""
        dev = qml.device("forest.numpy_wavefunction", wires=2)

        @qml.qnode(dev)
        def circuit():
            """Test QNode"""
            qml.Hadamard(wires=0)
            qml.PauliY(wires=0)
            return qml.expval(qml.PauliX(0))

        self.assertEqual(len(dev.program), 0)

        # construct and run the program
        circuit()
        self.assertEqual(len(dev.program), 2)
        self.assertEqual(str(dev.program), "H 0\nY 0\n")

    def test_wavefunction_args(self):
        """Test that the wavefunction plugin requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'wires'"):
            qml.device("forest.numpy_wavefunction")

    def test_hermitian_expectation(self, tol):
        """Test that an arbitrary Hermitian expectation value works"""
        dev = qml.device("forest.numpy_wavefunction", wires=1)

        @qml.qnode(dev)
        def circuit():
            """Test QNode"""
            qml.Hadamard(wires=0)
            qml.PauliY(wires=0)
            return qml.expval(qml.Hermitian(H, 0))

        out_state = 1j * np.array([-1, 1]) / np.sqrt(2)
        self.assertAllAlmostEqual(circuit(), np.vdot(out_state, H @ out_state), delta=tol)

    def test_qubit_unitary(self, tol):
        """Test that an arbitrary unitary operation works"""
        dev = qml.device("forest.numpy_wavefunction", wires=3)

        @qml.qnode(dev)
        def circuit():
            """Test QNode"""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.QubitUnitary(U2, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        out_state = U2 @ np.array([1, 0, 0, 1]) / np.sqrt(2)
        obs = np.kron(np.array([[1, 0], [0, -1]]), I)
        self.assertAllAlmostEqual(circuit(), np.vdot(out_state, obs @ out_state), delta=tol)

    def test_invalid_qubit_unitary(self):
        """Test that an invalid unitary operation is not allowed"""
        dev = qml.device("forest.numpy_wavefunction", wires=3)

        def circuit(Umat):
            """Test QNode"""
            qml.QubitUnitary(Umat, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit1 = qml.QNode(circuit, dev)
        with pytest.raises(ValueError, match="must be a square matrix"):
            circuit1(np.array([[0, 1]]))

        circuit1 = qml.QNode(circuit, dev)
        with pytest.raises(ValueError, match="must be unitary"):
            circuit1(np.array([[1, 1], [1, 1]]))

        circuit1 = qml.QNode(circuit, dev)
        with pytest.raises(ValueError, match=r"must be 2\^Nx2\^N"):
            circuit1(U)

    def test_one_qubit_wavefunction_circuit(self, tol):
        """Test that the wavefunction plugin provides correct result for simple circuit"""
        dev = qml.device("forest.numpy_wavefunction", wires=1)

        a = 0.543
        b = 0.123
        c = 0.987

        @qml.qnode(dev)
        def circuit(x, y, z):
            """Test QNode"""
            qml.BasisState(np.array([1]), wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        print(circuit(a, b, c))
        self.assertAlmostEqual(circuit(a, b, c), np.cos(a) * np.sin(b), delta=tol)

    def test_two_qubit_wavefunction_circuit(self, tol):
        """Test that the wavefunction plugin provides correct result for simple 2-qubit circuit,
        even when the number of wires > number of qubits."""
        dev = qml.device("forest.numpy_wavefunction", wires=3)

        a = 0.543
        b = 0.123
        c = 0.987
        theta = 0.6423

        @qml.qnode(dev)
        def circuit(w, x, y, z):
            """Test QNode"""
            qml.BasisState(np.array([0, 1]), wires=[0, 1])
            qml.Hadamard(wires=1)
            plf.CPHASE(w, 1, wires=[0, 1])
            qml.Rot(x, y, z, wires=0)
            plf.CPHASE(w, 3, wires=[0, 1])
            plf.CPHASE(w, 0, wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        self.assertAlmostEqual(
            circuit(theta, a, b, c), -np.sin(b / 2) ** 2 * np.sin(2 * theta), delta=tol
        )

    def test_nonzero_shots(self):
        """Test that the wavefunction plugin provides correct result for high shot number"""
        shots = 10 ** 2
        dev = qml.device("forest.numpy_wavefunction", wires=1, shots=shots)

        a = 0.543
        b = 0.123
        c = 0.987

        @qml.qnode(dev)
        def circuit(x, y, z):
            """Test QNode"""
            qml.BasisState(np.array([1]), wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        runs = []
        for _ in range(100):
            runs.append(circuit(a, b, c))

        expected_var = np.sqrt(1 / shots)
        print(np.mean(runs), np.cos(a) * np.sin(b))
        self.assertAlmostEqual(np.mean(runs), np.cos(a) * np.sin(b), delta=expected_var)
