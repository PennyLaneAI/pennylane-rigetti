"""
Unit tests for the QVM simulator device.
"""
import logging

import networkx as nx
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import Tensor
from pennylane.circuit_graph import CircuitGraph

from conftest import BaseTest
from conftest import I, Z, H, U, U2, test_operation_map, QVM_SHOTS

import pennylane_forest as plf

import pyquil

from flaky import flaky

log = logging.getLogger(__name__)

VALID_QPU_LATTICES = [qc for qc in pyquil.list_quantum_computers() if "qvm" not in qc]


class TestQVMBasic(BaseTest):
    """Unit tests for the QVM simulator."""
    # pylint: disable=protected-access

    def test_identity_expectation(self, shots, qvm, compiler):
        """Test that identity expectation value (i.e. the trace) is 1"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-qvm", shots=shots)

        O1 = qml.expval(qml.Identity(wires=[0]))
        O2 = qml.expval(qml.Identity(wires=[1]))

        circuit_graph = CircuitGraph([
                                       qml.RX(theta, wires=[0]),
                                       qml.RX(phi, wires=[1]),
                                       qml.CNOT(wires=[0, 1])
                                       ],
                                         [
                                        O1,
                                        O2
                                        ]
                                    )

        dev.apply(circuit_graph.operations, rotations=circuit_graph.diagonalizing_gates)

        dev.generate_samples()

        res = np.array([dev.expval(O1), dev.expval(O2)])

        # below are the analytic expectation values for this circuit (trace should always be 1)
        self.assertAllAlmostEqual(res, np.array([1, 1]), delta=3 / np.sqrt(shots))

    def test_pauliz_expectation(self, shots, qvm, compiler):
        """Test that PauliZ expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-qvm", shots=shots)

        O1 = qml.expval(qml.PauliZ(wires=[0]))
        O2 = qml.expval(qml.PauliZ(wires=[1]))

        circuit_graph = CircuitGraph([
                                       qml.RX(theta, wires=[0]),
                                       qml.RX(phi, wires=[1]),
                                       qml.CNOT(wires=[0, 1])
                                       ] +
                                         [O1, O2],
                                        {}
                                    )

        dev.apply(circuit_graph.operations, rotations=circuit_graph.diagonalizing_gates)

        dev.generate_samples()

        res = np.array([dev.expval(O1), dev.expval(O2)])

        # below are the analytic expectation values for this circuit
        self.assertAllAlmostEqual(
            res, np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]), delta=3 / np.sqrt(shots)
        )

    def test_paulix_expectation(self, shots, qvm, compiler):
        """Test that PauliX expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-qvm", shots=shots)
        O1 = qml.expval(qml.PauliX(wires=[0]))
        O2 = qml.expval(qml.PauliX(wires=[1]))

        circuit_graph = CircuitGraph([
                                       qml.RY(theta, wires=[0]),
                                       qml.RY(phi, wires=[1]),
                                       qml.CNOT(wires=[0, 1])
                                       ]
                                     +
                                     [O1, O2],
                                    {}
                                    )

        dev.apply(circuit_graph.operations, rotations=circuit_graph.diagonalizing_gates)

        dev.generate_samples()

        res = np.array([dev.expval(O1), dev.expval(O2)])
        # below are the analytic expectation values for this circuit
        self.assertAllAlmostEqual(
            res, np.array([np.sin(theta) * np.sin(phi), np.sin(phi)]), delta=3 / np.sqrt(shots)
        )

    def test_pauliy_expectation(self, shots, qvm, compiler):
        """Test that PauliY expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-qvm", shots=shots)
        O1 = qml.expval(qml.PauliY(wires=[0]))
        O2 = qml.expval(qml.PauliY(wires=[1]))

        circuit_graph = CircuitGraph([
                                       qml.RX(theta, wires=[0]),
                                       qml.RX(phi, wires=[1]),
                                       qml.CNOT(wires=[0, 1])
                                       ] +
                                         [O1, O2],
                                        {}
                                    )

        dev.apply(circuit_graph.operations, rotations=circuit_graph.diagonalizing_gates)

        dev.generate_samples()

        res = np.array([dev.expval(O1), dev.expval(O2)])

        # below are the analytic expectation values for this circuit
        self.assertAllAlmostEqual(
            res, np.array([0, -np.cos(theta) * np.sin(phi)]), delta=3 / np.sqrt(shots)
        )

    def test_hadamard_expectation(self, shots, qvm, compiler):
        """Test that Hadamard expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-qvm", shots=shots)
        O1 = qml.expval(qml.Hadamard(wires=[0]))
        O2 = qml.expval(qml.Hadamard(wires=[1]))

        circuit_graph = CircuitGraph([
                                       qml.RY(theta, wires=[0]),
                                       qml.RY(phi, wires=[1]),
                                       qml.CNOT(wires=[0, 1])
                                       ] +
                                         [O1, O2],
                                        {}
                                    )

        dev.apply(circuit_graph.operations, rotations=circuit_graph.diagonalizing_gates)

        dev.generate_samples()

        res = np.array([dev.expval(O1), dev.expval(O2)])

        # below are the analytic expectation values for this circuit
        expected = np.array(
            [np.sin(theta) * np.sin(phi) + np.cos(theta), np.cos(theta) * np.cos(phi) + np.sin(phi)]
        ) / np.sqrt(2)
        self.assertAllAlmostEqual(res, expected, delta=3 / np.sqrt(shots))

    def test_hermitian_expectation(self, shots, qvm, compiler):
        """Test that arbitrary Hermitian expectation values are correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-qvm", shots=shots)
        O1 = qml.expval(qml.Hermitian(H, wires=[0]))
        O2 = qml.expval(qml.Hermitian(H, wires=[1]))

        circuit_graph = CircuitGraph([
                                       qml.RY(theta, wires=[0]),
                                       qml.RY(phi, wires=[1]),
                                       qml.CNOT(wires=[0, 1])
                                       ] +
                                         [O1, O2],
                                        {}
                                    )

        dev.apply(circuit_graph.operations, rotations=circuit_graph.diagonalizing_gates)

        dev.generate_samples()

        res = np.array([dev.expval(O1), dev.expval(O2)])

        # below are the analytic expectation values for this circuit with arbitrary
        # Hermitian observable H
        a = H[0, 0]
        re_b = H[0, 1].real
        d = H[1, 1]
        ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
        expected = np.array([ev1, ev2])

        self.assertAllAlmostEqual(res, expected, delta=4 / np.sqrt(shots))

    def test_multi_qubit_hermitian_expectation(self, shots, qvm, compiler):
        """Test that arbitrary multi-qubit Hermitian expectation values are correct"""
        theta = 0.432
        phi = 0.123

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        dev = plf.QVMDevice(device="2q-qvm", shots=10 * shots)
        O1 = qml.expval(qml.Hermitian(A, wires=[0, 1]))

        circuit_graph = CircuitGraph([
                                       qml.RY(theta, wires=[0]),
                                       qml.RY(phi, wires=[1]),
                                       qml.CNOT(wires=[0, 1])
                                       ] +
                                         [O1],
                                        {}
                                    )

        dev.apply(circuit_graph.operations, rotations=circuit_graph.diagonalizing_gates)

        dev.generate_samples()

        res = np.array([dev.expval(O1)])
        # below is the analytic expectation value for this circuit with arbitrary
        # Hermitian observable A
        expected = 0.5 * (
            6 * np.cos(theta) * np.sin(phi)
            - np.sin(theta) * (8 * np.sin(phi) + 7 * np.cos(phi) + 3)
            - 2 * np.sin(phi)
            - 6 * np.cos(phi)
            - 6
        )

        self.assertAllAlmostEqual(res, expected, delta=4 / np.sqrt(shots))

    def test_var(self, shots, qvm, compiler):
        """Tests for variance calculation"""
        dev = plf.QVMDevice(device="2q-qvm", shots=shots)

        phi = 0.543
        theta = 0.6543

        O1 = qml.var(qml.PauliZ(wires=[0]))

        circuit_graph = CircuitGraph([
                                       qml.RX(phi, wires=[0]),
                                       qml.RY(theta, wires=[0]),
                                       ] +
                                         [O1],
                                        {}
                                    )

        # test correct variance for <Z> of a rotated state
        dev.apply(circuit_graph.operations, rotations=circuit_graph.diagonalizing_gates)

        dev.generate_samples()

        var = np.array([dev.var(O1)])
        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))

        self.assertAlmostEqual(var, expected, delta=3 / np.sqrt(shots))

    def test_var_hermitian(self, shots, qvm, compiler):
        """Tests for variance calculation using an arbitrary Hermitian observable"""
        dev = plf.QVMDevice(device="2q-qvm", shots=100 * shots)

        phi = 0.543
        theta = 0.6543

        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])

        O1 = qml.var(qml.Hermitian(A, wires=[0]))

        circuit_graph = CircuitGraph([
                                       qml.RX(phi, wires=[0]),
                                       qml.RY(theta, wires=[0]),
                                       ] +
                                         [O1],
                                        {}
                                    )

        # test correct variance for <A> of a rotated state
        dev.apply(circuit_graph.operations, rotations=circuit_graph.diagonalizing_gates)

        dev.generate_samples()

        var = np.array([dev.var(O1)])
        expected = 0.5 * (
            2 * np.sin(2 * theta) * np.cos(phi) ** 2
            + 24 * np.sin(phi) * np.cos(phi) * (np.sin(theta) - np.cos(theta))
            + 35 * np.cos(2 * phi)
            + 39
        )

        self.assertAlmostEqual(var, expected, delta=0.3)

    @pytest.mark.parametrize(
        "gate", plf.QVMDevice._operation_map
    )  # pylint: disable=protected-access
    def test_apply(self, gate, apply_unitary, shots, qvm, compiler):
        """Test the application of gates"""
        dev = plf.QVMDevice(device="3q-qvm", shots=shots)

        try:
            # get the equivalent pennylane operation class
            op = getattr(qml.ops, gate)
        except AttributeError:
            # get the equivalent pennylane-forest operation class
            op = getattr(plf, gate)

        # the list of wires to apply the operation to
        w = list(range(op.num_wires))

        obs = qml.expval(qml.PauliZ(0))
        if op.par_domain == "A":
            # the parameter is an array
            if gate == "QubitUnitary":
                p = np.array(U)
                w = [0]
                state = apply_unitary(U, 3)
            elif gate == "BasisState":
                p = np.array([1, 1, 1])
                state = np.array([0, 0, 0, 0, 0, 0, 0, 1])
                w = list(range(dev.num_wires))

            circuit_graph = CircuitGraph([
                                           op(p, wires=w)
                                           ] + [obs],
                                            {}
                                        )
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
            state = apply_unitary(O, 3)
            # Creating the circuit graph using a parametrized operation
            if p:
                circuit_graph = CircuitGraph([
                                               op(*p, wires=w)
                                               ] + [obs],
                                                {}
                                            )
            # Creating the circuit graph using an operation that take no parameters
            else:
                circuit_graph = CircuitGraph([
                                               op(wires=w)
                                               ] + [obs],
                                                {}
                                            )

        dev.apply(circuit_graph.operations, rotations=circuit_graph.diagonalizing_gates)

        dev.generate_samples()

        res = dev.expval(obs)
        expected = np.vdot(state, np.kron(np.kron(Z, I), I) @ state)

        # verify the device is now in the expected state
        # Note we have increased the tolerance here, since we are only
        # performing 1024 shots.
        self.assertAllAlmostEqual(res, expected, delta=3 / np.sqrt(shots))

    def test_sample_values(self, qvm, tol):
        """Tests if the samples returned by sample have
        the correct values
        """
        dev = plf.QVMDevice(device="1q-qvm", shots=10)

        O1 = qml.expval(qml.PauliZ(wires=[0]))

        circuit_graph = CircuitGraph([
                                       qml.RX(1.5708, wires=[0]),
                                       ] +
                                         [O1],
                                        {}
                                    )

        dev.apply(circuit_graph.operations, rotations=circuit_graph.diagonalizing_gates)

        dev.generate_samples()

        s1 = dev.sample(O1)

        # s1 should only contain 1 and -1
        self.assertAllAlmostEqual(s1**2, 1, delta=tol)
        self.assertAllAlmostEqual(s1, 1-2*dev._samples[:,0], delta=tol)

    def test_sample_values_hermitian(self, qvm, tol):
        """Tests if the samples of a Hermitian observable returned by sample have
        the correct values
        """
        theta = 0.543
        shots = 1000_000
        A = np.array([[1, 2j], [-2j, 0]])

        dev = plf.QVMDevice(device="1q-qvm", shots=shots)

        O1 = qml.sample(qml.Hermitian(A, wires=[0]))

        circuit_graph = CircuitGraph([
                                       qml.RX(theta, wires=[0]),
                                       ] +
                                         [O1],
                                     {}
                                    )

        dev.apply(circuit_graph.operations, rotations=circuit_graph.diagonalizing_gates)

        dev.generate_samples()

        s1 = dev.sample(O1)

        # s1 should only contain the eigenvalues of
        # the hermitian matrix
        eigvals = np.linalg.eigvalsh(A)
        assert np.allclose(sorted(list(set(s1))), sorted(eigvals), atol=tol, rtol=0)

        # the analytic mean is 2*sin(theta)+0.5*cos(theta)+0.5
        assert np.allclose(np.mean(s1), 2*np.sin(theta)+0.5*np.cos(theta)+0.5, atol=0.1, rtol=0)

        # the analytic variance is 0.25*(sin(theta)-4*cos(theta))^2
        assert np.allclose(np.var(s1), 0.25*(np.sin(theta)-4*np.cos(theta))**2, atol=0.1, rtol=0)

    def test_sample_values_hermitian_multi_qubit(self, qvm, tol):
        """Tests if the samples of a multi-qubit Hermitian observable returned by sample have
        the correct values
        """
        theta = 0.543
        shots = 100000

        A = np.array([
            [1,     2j,   1-2j, 0.5j  ],
            [-2j,   0,    3+4j, 1     ],
            [1+2j,  3-4j, 0.75, 1.5-2j],
            [-0.5j, 1,    1.5+2j, -1  ]
        ])

        dev = plf.QVMDevice(device="2q-qvm", shots=shots)

        O1 = qml.sample(qml.Hermitian(A, wires=[0, 1]))

        circuit_graph = CircuitGraph([
                                       qml.RX(theta, wires=[0]),
                                       qml.RY(2*theta, wires=[1]),
                                       qml.CNOT(wires=[0, 1]),
                                       ] +
                                         [O1],
                                     {}
                                    )

        dev.apply(circuit_graph.operations, rotations=circuit_graph.diagonalizing_gates)

        dev.generate_samples()

        s1 = dev.sample(O1)

        # s1 should only contain the eigenvalues of
        # the hermitian matrix
        eigvals = np.linalg.eigvalsh(A)
        assert np.allclose(sorted(list(set(s1))), sorted(eigvals), atol=tol, rtol=0)

        # make sure the mean matches the analytic mean
        expected = (88*np.sin(theta) + 24*np.sin(2*theta) - 40*np.sin(3*theta)
            + 5*np.cos(theta) - 6*np.cos(2*theta) + 27*np.cos(3*theta) + 6)/32
        assert np.allclose(np.mean(s1), expected, atol=0.1, rtol=0)

    def test_raise_error_if_analytic_true(self, shots):
        """Test that instantiating a QVMDevice in analytic=True mode raises an error"""
        with pytest.raises(
            ValueError, match="QVM device cannot be run in analytic=True mode."
        ):
            dev = plf.QVMDevice(device="2q-qvm", shots=shots, analytic=True)


class TestQVMIntegration(BaseTest):
    """Test the QVM simulator works correctly from the PennyLane frontend."""

    # pylint: disable=no-self-use

    def test_load_qvm_device(self, qvm):
        """Test that the QVM device loads correctly"""
        dev = qml.device("forest.qvm", device="2q-qvm")
        self.assertEqual(dev.num_wires, 2)
        self.assertEqual(dev.shots, 1024)
        self.assertEqual(dev.short_name, "forest.qvm")

    def test_load_qvm_device_from_topology(self, qvm):
        """Test that the QVM device, from an input topology, loads correctly"""
        topology = nx.complete_graph(2)
        dev = qml.device("forest.qvm", device=topology)
        self.assertEqual(dev.num_wires, 2)
        self.assertEqual(dev.shots, 1024)
        self.assertEqual(dev.short_name, "forest.qvm")

    def test_load_virtual_qpu_device(self, qvm):
        """Test that the QPU simulators load correctly"""
        qml.device("forest.qvm", device=np.random.choice(VALID_QPU_LATTICES))

    def test_incorrect_qc_name(self):
        """Test that exception is raised if name is incorrect"""
        with pytest.raises(
            ValueError, match="QVM device string does not indicate the number of qubits"
        ):
            qml.device("forest.qvm", device="Aspen-1-B")

    def test_incorrect_qc_type(self):
        """Test that exception is raised device is not a string or graph"""
        with pytest.raises(ValueError, match="Required argument device must be a string"):
            qml.device("forest.qvm", device=3)

    def test_qvm_args(self):
        """Test that the QVM plugin requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device("forest.qvm")

        with pytest.raises(ValueError, match="Number of shots must be a positive integer"):
            qml.device("forest.qvm", "2q-qvm", shots=0)

    def test_qubit_unitary(self, shots, qvm, compiler):
        """Test that an arbitrary unitary operation works"""
        dev1 = qml.device("forest.qvm", device="3q-qvm", shots=shots)
        dev2 = qml.device("forest.qvm", device="9q-square-qvm", shots=shots)

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

    @flaky(max_runs=10, min_passes=1)
    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(VALID_QPU_LATTICES)])
    def test_one_qubit_wavefunction_circuit(self, device, qvm, compiler):
        """Test that the wavefunction plugin provides correct result for simple circuit"""
        shots = 100000
        dev = qml.device("forest.qvm", device=device, shots=QVM_SHOTS)

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

        self.assertAlmostEqual(circuit(a, b, c), np.cos(a) * np.sin(b), delta=3 / np.sqrt(shots))

    @flaky(max_runs=10, min_passes=1)
    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(VALID_QPU_LATTICES)])
    def test_2q_gate_pauliz_identity_tensor(self, device, qvm, compiler):
        dev = qml.device("forest.qvm", device=device, shots=QVM_SHOTS)

        @qml.qnode(dev)
        def circuit():
            qml.RY(np.pi/2, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.Identity(1))

        assert np.allclose(circuit(), 0.0, atol=2e-2)

    @flaky(max_runs=10, min_passes=1)
    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(VALID_QPU_LATTICES)])
    def test_2q_gate_pauliz_pauliz_tensor(self, device, qvm, compiler):
        dev = qml.device("forest.qvm", device=device, shots=QVM_SHOTS)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert np.allclose(circuit(), 1.0, atol=2e-2)
