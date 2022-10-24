"""
Unit tests for the wavefunction simulator device.
"""
import logging

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.wires import Wires

from conftest import BaseTest
from conftest import I, Z, H, U, U2, SWAP, CNOT, U_toffoli, H, test_operation_map

import pennylane_forest as plf


log = logging.getLogger(__name__)


class TestWavefunctionBasic(BaseTest):
    """Unit tests for the wavefunction simulator."""

    def test_var(self, tol, qvm):
        """Tests for variance calculation"""
        dev = plf.WavefunctionDevice(wires=2)

        phi = 0.543
        theta = 0.6543

        with qml.tape.QuantumTape() as tape:
            qml.RX(phi, wires=[0])
            qml.RY(theta, wires=[0])
            O = qml.var(qml.PauliZ(wires=[0]))

        # test correct variance for <Z> of a rotated state
        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)

        var = dev.var(O)
        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))

        self.assertAlmostEqual(var, expected, delta=tol)

    def test_var_hermitian(self, tol, qvm):
        """Tests for variance calculation using an arbitrary Hermitian observable"""
        dev = plf.WavefunctionDevice(wires=2)

        phi = 0.543
        theta = 0.6543

        # test correct variance for <H> of a rotated state
        H = np.array([[4, -1 + 6j], [-1 - 6j, 2]])

        with qml.tape.QuantumTape() as tape:
            qml.RX(phi, wires=[0])
            qml.RY(theta, wires=[0])
            O = qml.var(qml.Hermitian(H, wires=[0]))

        # test correct variance for <Z> of a rotated state
        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)

        var = dev.var(O)
        expected = 0.5 * (
            2 * np.sin(2 * theta) * np.cos(phi) ** 2
            + 24 * np.sin(phi) * np.cos(phi) * (np.sin(theta) - np.cos(theta))
            + 35 * np.cos(2 * phi)
            + 39
        )

        self.assertAlmostEqual(var, expected, delta=tol)

    @pytest.mark.parametrize(
        "op",
        [
            qml.QubitUnitary(np.array(U), wires=0),
            qml.BasisState(np.array([1, 1, 1]), wires=list(range(3))),
            qml.PauliX(wires=0),
            qml.PauliY(wires=0),
            qml.PauliZ(wires=0),
            qml.S(wires=0),
            qml.T(wires=0),
            qml.RX(0.432, wires=0),
            qml.RY(0.432, wires=0),
            qml.RZ(0.432, wires=0),
            qml.Hadamard(wires=0),
            qml.Rot(0.432, 2, 0.324, wires=0),
            qml.Toffoli(wires=[0, 1, 2]),
            qml.SWAP(wires=[0, 1]),
            qml.CSWAP(wires=[0, 1, 2]),
            qml.CZ(wires=[0, 1]),
            qml.CNOT(wires=[0, 1]),
            qml.PhaseShift(0.432, wires=0),
            qml.CSWAP(wires=[0, 1, 2]),
            plf.CPHASE(0.432, 2, wires=[0, 1]),
            qml.ISWAP(wires=[0, 1]),
            qml.PSWAP(0.432, wires=[0, 1]),
        ],
    )
    def test_apply(self, op, apply_unitary, tol):
        """Test the application of gates to a state"""
        dev = plf.WavefunctionDevice(wires=3)

        obs = qml.expval(qml.PauliZ(0))

        if op.name == "QubitUnitary":
            state = apply_unitary(U, 3)
        elif op.name == "BasisState":
            state = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        elif op.name == "CPHASE":
            state = apply_unitary(test_operation_map["CPHASE"](0.432, 2), 3)
        elif op.name == "ISWAP":
            state = apply_unitary(test_operation_map["ISWAP"], 3)
        elif op.name == "PSWAP":
            state = apply_unitary(test_operation_map["PSWAP"](0.432), 3)
        else:
            state = apply_unitary(qml.matrix(op), 3)

        with qml.tape.QuantumTape() as tape:
            qml.apply(op)
            obs

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)

        # verify the device is now in the expected state
        self.assertAllAlmostEqual(dev.state, state, delta=tol)

    def test_sample_values(self, tol):
        """Tests if the samples returned by sample have
        the correct values
        """
        dev = plf.WavefunctionDevice(wires=1, shots=10)
        theta = 1.5708

        circuit_operations = []

        O = qml.sample(qml.PauliZ(0))

        observables = [O]

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            qml.sample(qml.PauliZ(0))

        dev.apply(tape._ops, rotations=tape.diagonalizing_gates)
        dev._samples = dev.generate_samples()
        s1 = dev.sample(O.obs)

        # s1 should only contain 1 and -1
        self.assertAllAlmostEqual(s1 ** 2, 1, delta=tol)

    def test_sample_values_hermitian(self, tol):
        """Tests if the samples of a Hermitian observable returned by sample have
        the correct values
        """
        dev = plf.WavefunctionDevice(wires=1, shots=1_000_000)
        theta = 0.543

        A = np.array([[1, 2j], [-2j, 0]])

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            O = qml.sample(qml.Hermitian(A, wires=[0]))

        # test correct variance for <Z> of a rotated state
        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)

        dev._samples = dev.generate_samples()

        s1 = dev.sample(O.obs)

        # s1 should only contain the eigenvalues of
        # the hermitian matrix
        eigvals = np.linalg.eigvalsh(A)
        assert np.allclose(sorted(list(set(s1))), sorted(eigvals), atol=tol, rtol=0)

        # the analytic mean is 2*sin(theta)+0.5*cos(theta)+0.5
        assert np.allclose(
            np.mean(s1), 2 * np.sin(theta) + 0.5 * np.cos(theta) + 0.5, atol=0.1, rtol=0
        )

        # the analytic variance is 0.25*(sin(theta)-4*cos(theta))^2
        assert np.allclose(
            np.var(s1), 0.25 * (np.sin(theta) - 4 * np.cos(theta)) ** 2, atol=0.1, rtol=0
        )

    def test_sample_values_hermitian_multi_qubit(self, tol):
        """Tests if the samples of a multi-qubit Hermitian observable returned by sample have
        the correct values
        """
        shots = 1_000_000
        dev = plf.WavefunctionDevice(wires=2, shots=shots)
        theta = 0.543

        A = np.array(
            [
                [1, 2j, 1 - 2j, 0.5j],
                [-2j, 0, 3 + 4j, 1],
                [1 + 2j, 3 - 4j, 0.75, 1.5 - 2j],
                [-0.5j, 1, 1.5 + 2j, -1],
            ]
        )

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0]),
            qml.RY(2 * theta, wires=[1]),
            qml.CNOT(wires=[0, 1]),
            O = qml.sample(qml.Hermitian(A, wires=[0, 1]))

        # test correct variance for <Z> of a rotated state
        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)

        dev._samples = dev.generate_samples()

        s1 = dev.sample(O.obs)

        # s1 should only contain the eigenvalues of
        # the hermitian matrix
        eigvals = np.linalg.eigvalsh(A)
        assert np.allclose(sorted(list(set(s1))), sorted(eigvals), atol=tol, rtol=0)

        # make sure the mean matches the analytic mean
        expected = (
            88 * np.sin(theta)
            + 24 * np.sin(2 * theta)
            - 40 * np.sin(3 * theta)
            + 5 * np.cos(theta)
            - 6 * np.cos(2 * theta)
            + 27 * np.cos(3 * theta)
            + 6
        ) / 32
        assert np.allclose(np.mean(s1), expected, atol=0.1, rtol=0)


class TestWavefunctionIntegration(BaseTest):
    """Test the wavefunction simulator works correctly from the PennyLane frontend."""

    # pylint:disable=no-self-use

    def test_load_wavefunction_device(self):
        """Test that the wavefunction device loads correctly"""
        dev = qml.device("forest.wavefunction", wires=2)
        self.assertEqual(dev.num_wires, 2)
        self.assertEqual(dev.shots, None)
        self.assertEqual(dev.short_name, "forest.wavefunction")

    def test_program_property(self, qvm, compiler):
        """Test that the program property works as expected"""
        dev = qml.device("forest.wavefunction", wires=2)

        @qml.qnode(dev)
        def circuit():
            """Test QNode"""
            qml.Hadamard(wires=0)
            qml.PauliY(wires=0)
            return qml.expval(qml.PauliX(0))

        self.assertEqual(len(dev.program), 0)

        # construct and run the program
        circuit()

        # Consider the gates used for diagonalization as well
        self.assertEqual(len(dev.program), 2 + 1)
        self.assertEqual(str(dev.program), "H 0\nY 0\nH 0\n")

    def test_wavefunction_args(self):
        """Test that the wavefunction plugin requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'wires'"):
            qml.device("forest.wavefunction")

    def test_hermitian_expectation(self, tol, qvm, compiler):
        """Test that an arbitrary Hermitian expectation value works"""
        dev = qml.device("forest.wavefunction", wires=1)

        @qml.qnode(dev)
        def circuit():
            """Test QNode"""
            qml.Hadamard(wires=0)
            qml.PauliY(wires=0)
            return qml.expval(qml.Hermitian(H, 0))

        out_state = 1j * np.array([-1, 1]) / np.sqrt(2)
        self.assertAllAlmostEqual(circuit(), np.vdot(out_state, H @ out_state), delta=tol)

    def test_qubit_unitary(self, tol, qvm, compiler):
        """Test that an arbitrary unitary operation works"""
        dev = qml.device("forest.wavefunction", wires=3)

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
        dev = qml.device("forest.wavefunction", wires=3)
        dev.shots = 1

        def circuit(Umat):
            """Test QNode"""
            qml.QubitUnitary(Umat, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit1 = qml.QNode(circuit, dev)
        with pytest.raises(ValueError, match="Input unitary must be of shape"):
            circuit1(np.array([[0, 1]]))

    def test_one_qubit_wavefunction_circuit(self, tol, qvm, compiler):
        """Test that the wavefunction plugin provides correct result for simple circuit"""
        dev = qml.device("forest.wavefunction", wires=1)

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

        self.assertAlmostEqual(circuit(a, b, c), np.cos(a) * np.sin(b), delta=tol)

    def test_two_qubit_wavefunction_circuit(self, tol, qvm, compiler):
        """Test that the wavefunction plugin provides correct result for simple 2-qubit circuit,
        even when the number of wires > number of qubits."""
        dev = qml.device("forest.wavefunction", wires=3)

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

    def test_nonzero_shots(self, qvm, compiler):
        """Test that the wavefunction plugin provides correct result for high shot number"""
        shots = 10 ** 2
        dev = qml.device("forest.wavefunction", wires=1, shots=shots)

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
        self.assertAlmostEqual(np.mean(runs), np.cos(a) * np.sin(b), delta=expected_var)

class TestExpandState(BaseTest):
    """Test the expand_state method"""

    def test_expand_state(self):
        """Test that a multi-qubit state is correctly expanded for a N-qubit device"""
        dev = plf.WavefunctionDevice(wires=3)

        # expand a two qubit state to the 3 qubit device
        dev._state = np.array([0, 1, 1, 0]) / np.sqrt(2)
        dev._active_wires = Wires([0, 2])
        dev.expand_state()
        self.assertAllEqual(dev.state, np.array([0, 1, 0, 0, 1, 0, 0, 0]) / np.sqrt(2))

        # expand a three qubit state to the 3 qubit device
        dev._state = np.array([0, 1, 1, 0, 0, 1, 1, 0]) / 2
        dev._active_wires = Wires([0, 1, 2])
        dev.expand_state()
        self.assertAllEqual(dev.state, np.array([0, 1, 1, 0, 0, 1, 1, 0]) / 2)


    @pytest.mark.parametrize("bitstring, dec", [([1], 1), ([0,0,1], 1), ([0,1,0], 2), ([1,1,1], 7)])
    def test_bit2dec(self, bitstring, dec):
        """Test that the bit2dec method produces the correct decimal values for
        bitstrings"""
        assert plf.WavefunctionDevice.bit2dec(bitstring) == dec

    @pytest.mark.parametrize("num_wires", [3,4] )
    @pytest.mark.parametrize("wire_idx1, wire_idx2", [(0, 0), (1, 0), (1, 1)] )
    def test_expanded_matches_default_qubit(self, wire_idx1, wire_idx2, num_wires):
        """Test that the statevector is expanded correctly by checking the
        output of a QNode with default.qubit."""
        dev_default_qubit = qml.device('default.qubit', wires=num_wires)
        dev_wavefunction = qml.device('forest.wavefunction', wires=num_wires)

        def circuit():
            qml.RY(np.pi/2, wires=[wire_idx1 + 1])
            qml.RY(np.pi, wires=[wire_idx2 + 0])
            qml.CNOT(wires=[wire_idx1 + 1, wire_idx2 + 0])
            qml.CNOT(wires=[wire_idx1 + 1, wire_idx2 + 0])
            return qml.probs(range(num_wires))

        qnode1 = qml.QNode(circuit, dev_wavefunction)
        qnode2 = qml.QNode(circuit, dev_default_qubit)
        assert np.allclose(qnode1(), qnode2())
