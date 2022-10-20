"""
Unit tests for the QVM simulator device.
"""
import logging

import networkx as nx
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.wires import Wires

from pyquil.api._quantum_computer import QuantumComputer

from conftest import BaseTest
from conftest import I, Z, H, U, U2, test_operation_map, QVM_SHOTS

import pennylane_forest as plf

import pyquil

from flaky import flaky

log = logging.getLogger(__name__)

TEST_QPU_LATTICES = ["4q-qvm"]


compiled_program = (
    "DECLARE ro BIT[2]\n"
    'PRAGMA INITIAL_REWIRING "PARTIAL"\n'
    "RZ(0.432) 1\n"
    "CZ 1 0\n"
    "MEASURE 1 ro[0]\n"
    "MEASURE 0 ro[1]\n"
    "HALT\n"
)


class TestQVMBasic(BaseTest):
    """Unit tests for the QVM simulator."""

    # pylint: disable=protected-access

    def test_identity_expectation(self, shots, qvm, compiler):
        """Test that identity expectation value (i.e. the trace) is 1"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-qvm", shots=shots)

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            O1 = qml.expval(qml.Identity(wires=[0]))
            O2 = qml.expval(qml.Identity(wires=[1]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)

        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1.obs), dev.expval(O2.obs)])

        # below are the analytic expectation values for this circuit (trace should always be 1)
        self.assertAllAlmostEqual(res, np.array([1, 1]), delta=3 / np.sqrt(shots))

    def test_pauliz_expectation(self, shots, qvm, compiler):
        """Test that PauliZ expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-qvm", shots=shots)

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            O1 = qml.expval(qml.PauliZ(wires=[0]))
            O2 = qml.expval(qml.PauliZ(wires=[1]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)
        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1.obs), dev.expval(O2.obs)])

        # below are the analytic expectation values for this circuit
        self.assertAllAlmostEqual(
            res, np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]), delta=3 / np.sqrt(shots)
        )

    def test_paulix_expectation(self, shots, qvm, compiler):
        """Test that PauliX expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-qvm", shots=shots)

        with qml.tape.QuantumTape() as tape:
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            O1 = qml.expval(qml.PauliX(wires=[0]))
            O2 = qml.expval(qml.PauliX(wires=[1]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)
        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1.obs), dev.expval(O2.obs)])
        # below are the analytic expectation values for this circuit
        self.assertAllAlmostEqual(
            res, np.array([np.sin(theta) * np.sin(phi), np.sin(phi)]), delta=3 / np.sqrt(shots)
        )

    def test_pauliy_expectation(self, shots, qvm, compiler):
        """Test that PauliY expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-qvm", shots=shots)

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            O1 = qml.expval(qml.PauliY(wires=[0]))
            O2 = qml.expval(qml.PauliY(wires=[1]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)
        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1.obs), dev.expval(O2.obs)])

        # below are the analytic expectation values for this circuit
        self.assertAllAlmostEqual(
            res, np.array([0, -np.cos(theta) * np.sin(phi)]), delta=3 / np.sqrt(shots)
        )

    def test_hadamard_expectation(self, shots, qvm, compiler):
        """Test that Hadamard expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-qvm", shots=shots)

        with qml.tape.QuantumTape() as tape:
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            O1 = qml.expval(qml.Hadamard(wires=[0]))
            O2 = qml.expval(qml.Hadamard(wires=[1]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)
        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1.obs), dev.expval(O2.obs)])

        # below are the analytic expectation values for this circuit
        expected = np.array(
            [np.sin(theta) * np.sin(phi) + np.cos(theta), np.cos(theta) * np.cos(phi) + np.sin(phi)]
        ) / np.sqrt(2)
        self.assertAllAlmostEqual(res, expected, delta=3 / np.sqrt(shots))

    @flaky(max_runs=10, min_passes=3)
    def test_hermitian_expectation(self, shots, qvm, compiler):
        """Test that arbitrary Hermitian expectation values are correct.

        As the results coming from the qvm are stochastic, a constraint of 3 out of 5 runs was added.
        """

        theta = 0.432
        phi = 0.123

        dev = plf.QVMDevice(device="2q-qvm", shots=shots)

        with qml.tape.QuantumTape() as tape:
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            O1 = qml.expval(qml.Hermitian(H, wires=[0]))
            O2 = qml.expval(qml.Hermitian(H, wires=[1]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)
        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1.obs), dev.expval(O2.obs)])

        # below are the analytic expectation values for this circuit with arbitrary
        # Hermitian observable H
        a = H[0, 0]
        re_b = H[0, 1].real
        d = H[1, 1]
        ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
        expected = np.array([ev1, ev2])

        self.assertAllAlmostEqual(res, expected, delta=4 / np.sqrt(shots))

    def test_multi_qubit_hermitian_expectation(self, shots, execution_timeout, qvm, compiler):
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

        dev = plf.QVMDevice(device="2q-qvm", shots=10 * shots, execution_timeout=execution_timeout)

        with qml.tape.QuantumTape() as tape:
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            O1 = qml.expval(qml.Hermitian(A, wires=[0, 1]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)
        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1.obs)])
        # below is the analytic expectation value for this circuit with arbitrary
        # Hermitian observable A
        expected = 0.5 * (
            6 * np.cos(theta) * np.sin(phi)
            - np.sin(theta) * (8 * np.sin(phi) + 7 * np.cos(phi) + 3)
            - 2 * np.sin(phi)
            - 6 * np.cos(phi)
            - 6
        )

        self.assertAllAlmostEqual(res, expected, delta=5 / np.sqrt(shots))

    def test_var(self, shots, qvm, compiler):
        """Tests for variance calculation"""
        dev = plf.QVMDevice(device="2q-qvm", shots=shots)

        phi = 0.543
        theta = 0.6543

        with qml.tape.QuantumTape() as tape:
            qml.RX(phi, wires=[0])
            qml.RY(theta, wires=[0])
            O1 = qml.var(qml.PauliZ(wires=[0]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)
        dev._samples = dev.generate_samples()

        var = np.array([dev.var(O1.obs)])
        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))

        self.assertAlmostEqual(var, expected, delta=3 / np.sqrt(shots))

    def test_var_hermitian(self, shots, execution_timeout, qvm, compiler):
        """Tests for variance calculation using an arbitrary Hermitian observable"""
        dev = plf.QVMDevice(device="2q-qvm", shots=100 * shots, execution_timeout=execution_timeout)

        phi = 0.543
        theta = 0.6543

        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])

        with qml.tape.QuantumTape() as tape:
            qml.RX(phi, wires=[0])
            qml.RY(theta, wires=[0])
            O1 = qml.var(qml.Hermitian(A, wires=[0]))

        # test correct variance for <A> of a rotated state
        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)
        dev._samples = dev.generate_samples()

        var = np.array([dev.var(O1.obs)])
        expected = 0.5 * (
            2 * np.sin(2 * theta) * np.cos(phi) ** 2
            + 24 * np.sin(phi) * np.cos(phi) * (np.sin(theta) - np.cos(theta))
            + 35 * np.cos(2 * phi)
            + 39
        )

        self.assertAlmostEqual(var, expected, delta=0.3)

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
    def test_apply(self, op, apply_unitary, shots, qvm, compiler):
        """Test the application of gates to a state"""
        dev = plf.QVMDevice(device="3q-qvm", shots=shots, parametric_compilation=False)

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

        dev._samples = dev.generate_samples()

        res = dev.expval(obs.obs)
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

        with qml.tape.QuantumTape() as tape:
            qml.RX(1.5708, wires=[0])
            O1 = qml.expval(qml.PauliZ(wires=[0]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)

        dev._samples = dev.generate_samples()

        s1 = dev.sample(O1.obs)

        # s1 should only contain 1 and -1
        self.assertAllAlmostEqual(s1 ** 2, 1, delta=tol)
        self.assertAllAlmostEqual(s1, 1 - 2 * dev._samples[:, 0], delta=tol)

    def test_sample_values_hermitian(self, qvm, execution_timeout, tol):
        """Tests if the samples of a Hermitian observable returned by sample have
        the correct values
        """
        theta = 0.543
        shots = 1_000_000
        A = np.array([[1, 2j], [-2j, 0]])

        dev = plf.QVMDevice(device="1q-qvm", shots=shots, execution_timeout=execution_timeout)

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            O1 = qml.sample(qml.Hermitian(A, wires=[0]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)

        dev._samples = dev.generate_samples()

        s1 = dev.sample(O1.obs)

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

    def test_sample_values_hermitian_multi_qubit(self, qvm, execution_timeout, tol):
        """Tests if the samples of a multi-qubit Hermitian observable returned by sample have
        the correct values
        """
        theta = 0.543
        shots = 100_000

        A = np.array(
            [
                [1, 2j, 1 - 2j, 0.5j],
                [-2j, 0, 3 + 4j, 1],
                [1 + 2j, 3 - 4j, 0.75, 1.5 - 2j],
                [-0.5j, 1, 1.5 + 2j, -1],
            ]
        )

        dev = plf.QVMDevice(device="2q-qvm", shots=shots, execution_timeout=execution_timeout)

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            qml.RY(2 * theta, wires=[1])
            qml.CNOT(wires=[0, 1])
            O1 = qml.sample(qml.Hermitian(A, wires=[0, 1]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)

        dev._samples = dev.generate_samples()

        s1 = dev.sample(O1.obs)

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

    def test_wires_argument(self):
        """Test that the wires argument gets processed correctly."""

        dev_no_wires = plf.QVMDevice(device="2q-qvm", shots=5)
        assert dev_no_wires.wires == Wires(range(2))

        with pytest.raises(ValueError, match="Device has a fixed number of"):
            plf.QVMDevice(device="2q-qvm", shots=5, wires=1000)

        dev_iterable_wires = plf.QVMDevice(device="2q-qvm", shots=5, wires=range(2))
        assert dev_iterable_wires.wires == Wires(range(2))

        with pytest.raises(ValueError, match="Device has a fixed number of"):
            plf.QVMDevice(device="2q-qvm", shots=5, wires=range(1000))

    @pytest.mark.parametrize("shots", list(range(0, -10, -1)))
    def test_raise_error_if_shots_is_not_positive(self, shots):
        """Test that instantiating a QVMDevice if the number of shots is not a postivie
        integer raises an error"""
        with pytest.raises(ValueError, match="Number of shots must be a positive integer."):
            dev = plf.QVMDevice(device="2q-qvm", shots=shots)

    def test_raise_error_if_shots_is_none(self, shots):
        """Test that instantiating a QVMDevice to be used for analytic computations raises an error"""
        with pytest.raises(ValueError, match="QVM device cannot be used for analytic computations."):
            dev = plf.QVMDevice(device="2q-qvm", shots=None)

    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(TEST_QPU_LATTICES)])
    def test_timeout_set_correctly(self, shots, device):
        """Test that the timeout attrbiute for the QuantumComputer stored by the QVMDevice
        is set correctly when passing a value as keyword argument"""
        dev = plf.QVMDevice(device=device, shots=shots, compiler_timeout=100, execution_timeout=101)
        assert dev.qc.compiler._timeout == 100
        assert dev.qc.qam._qvm_client.timeout == 101

    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(TEST_QPU_LATTICES)])
    def test_timeout_default(self, shots, device):
        """Test that the timeout attrbiute for the QuantumComputer stored by the QVMDevice
        is set correctly when passing a value as keyword argument"""
        dev = plf.QVMDevice(device=device, shots=shots)
        qc = pyquil.get_qc(device, as_qvm=True)

        # Check that the timeouts are equal (it has not been changed as a side effect of
        # instantiation
        assert dev.qc.compiler._timeout == qc.compiler._timeout
        assert dev.qc.qam._qvm_client.timeout == qc.qam._qvm_client.timeout

    def test_compiled_program_stored(self, qvm, monkeypatch):
        """Test that QVM device stores the latest compiled program."""
        dev = qml.device("forest.qvm", device="2q-qvm")

        assert dev.compiled_program is None

        theta = 0.432
        phi = 0.123

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            O1 = qml.expval(qml.Identity(wires=[0]))
            O2 = qml.expval(qml.Identity(wires=[1]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)

        dev.generate_samples()

        assert dev.compiled_program is not None

    def test_stored_compiled_program_correct(self, qvm, monkeypatch):
        """Test that QVM device stores the latest compiled program."""
        dev = qml.device("forest.qvm", device="2q-qvm")

        assert dev.compiled_program is None

        theta = 0.432

        with qml.tape.QuantumTape() as tape:
            qml.RZ(theta, wires=[0])
            qml.CZ(wires=[0, 1])
            O1 = qml.expval(qml.PauliZ(wires=[0]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)

        dev.generate_samples()

        assert dev.compiled_program == compiled_program


class TestParametricCompilation(BaseTest):
    """Test that parametric compilation works fine and the same program only compiles once."""

    def test_compiled_program_was_stored_in_dict(self, qvm, mock_qvm, monkeypatch):
        """Test that QVM device stores the compiled program correctly in a dictionary"""
        dev = qml.device("forest.qvm", device="2q-qvm")
        theta = 0.432
        phi = 0.123

        with qml.tape.QuantumTape() as tape:
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            O1 = qml.expval(qml.Identity(wires=[0]))
            O2 = qml.expval(qml.Identity(wires=[1]))

        dev.apply(tape.operations, rotations=tape.diagonalizing_gates)

        dev._circuit_hash = tape.graph.hash
        dev.generate_samples()

        assert dev.circuit_hash in dev._compiled_program_dict
        assert len(dev._compiled_program_dict.items()) == 1
        assert mock_qvm.compile.call_count == 1

        # Testing that the compile() method was not called
        # Calling generate_samples with unchanged hash
        for _ in range(6):
            dev.generate_samples()

            assert dev.circuit_hash in dev._compiled_program_dict
            assert len(dev._compiled_program_dict.items()) == 1
            assert mock_qvm.compile.call_count == 1

    def test_parametric_compilation_with_numeric_and_symbolic_queue(self, mock_qvm, execution_timeout):
        """Tests that a program containing numeric and symbolic variables as
        well is only compiled once."""
        dev = qml.device("forest.qvm", device="2q-qvm", execution_timeout=execution_timeout)

        param1 = np.array(1, requires_grad=False)
        param2 = np.array(2, requires_grad=True)

        dev._circuit_hash = None

        number_of_runs = 10

        first = True

        for run_idx in range(number_of_runs):

            with qml.tape.QuantumTape() as tape:
                qml.RX(param1, wires=[0])
                qml.RX(param2, wires=[1])

            dev.apply(tape.operations, rotations=tape.diagonalizing_gates)

            if first:
                dev._circuit_hash = tape.graph.hash
                first = False
            else:
                # Check that we are still producing the same circuit hash
                assert dev._circuit_hash == tape.graph.hash

            dev.generate_samples()

        assert len(dev._compiled_program_dict.items()) == 1
        assert mock_qvm.compile.call_count == 1

    def test_apply_qubitstatesvector_raises_an_error_if_not_first(self):
        """Test that there is an error raised when the QubitStateVector is not
        applied as the first operation."""
        dev = qml.device("forest.qvm", device="2q-qvm", parametric_compilation=True)

        operation = qml.QubitStateVector(np.array([1, 0]), wires=list(range(2)))
        queue = [qml.PauliX(0), operation]
        with pytest.raises(
            qml.DeviceError,
            match="Operation {} cannot be used after other Operations have already been applied".format(
                operation.name
            ),
        ):
            dev.apply(queue)


class TestQVMIntegration(BaseTest):
    """Test the QVM simulator works correctly from the PennyLane frontend."""

    # pylint: disable=no-self-use

    def test_load_qvm_device(self, qvm):
        """Test that the QVM device loads correctly"""
        dev = qml.device("forest.qvm", device="2q-qvm")
        self.assertEqual(dev.num_wires, 2)
        self.assertEqual(dev.shots, 1000)
        self.assertEqual(dev.short_name, "forest.qvm")

    def test_load_qvm_device_from_topology(self, qvm):
        """Test that the QVM device, from an input topology, loads correctly"""
        topology = nx.complete_graph(2)
        dev = qml.device("forest.qvm", device=topology)
        self.assertEqual(dev.num_wires, 2)
        self.assertEqual(dev.shots, 1000)
        self.assertEqual(dev.short_name, "forest.qvm")

    def test_load_virtual_qpu_device(self, qvm):
        """Test that the QPU simulators load correctly"""
        qml.device("forest.qvm", device=np.random.choice(TEST_QPU_LATTICES))

    def test_qvm_args(self):
        """Test that the QVM plugin requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device("forest.qvm")

        with pytest.raises(ValueError, match="Number of shots must be a positive integer"):
            qml.device("forest.qvm", "2q-qvm", shots=0)

    def test_qubit_unitary(self, shots, compiler_timeout, execution_timeout, qvm, compiler):
        """Test that an arbitrary unitary operation works"""
        dev1 = qml.device("forest.qvm", device="3q-qvm", shots=shots, compiler_timeout=compiler_timeout, execution_timeout=execution_timeout, parametric_compilation=False)
        dev2 = qml.device("forest.qvm", device="9q-square-qvm", shots=shots, compiler_timeout=compiler_timeout, execution_timeout=execution_timeout, parametric_compilation=False)

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

    @flaky(max_runs=10, min_passes=2)
    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(TEST_QPU_LATTICES)])
    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_one_qubit_wavefunction_circuit(self, device, qvm, compiler, requires_grad):
        """Test that the wavefunction plugin provides correct result for simple circuit.

        As the results coming from the qvm are stochastic, a constraint of 2 out of 5 runs was added.
        """
        shots = 100_000
        dev = qml.device("forest.qvm", device=device, shots=QVM_SHOTS)

        a = 0.543
        b = 0.123
        c = 0.987

        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.BasisState(np.array([1], requires_grad=requires_grad), wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        self.assertAlmostEqual(circuit(a, b, c), np.cos(a) * np.sin(b), delta=3 / np.sqrt(shots))

    @flaky(max_runs=10, min_passes=3)
    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(TEST_QPU_LATTICES)])
    def test_2q_gate(self, device, qvm, compiler):
        """Test that the two qubit gate with the PauliZ observable works correctly.

        As the results coming from the qvm are stochastic, a constraint of 3 out of 5 runs was added.
        """
        dev = qml.device("forest.qvm", device=device, shots=QVM_SHOTS)

        @qml.qnode(dev)
        def circuit():
            qml.RY(np.pi / 2, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.Identity(1))

        assert np.allclose(circuit(), 0.0, atol=2e-2)

    @flaky(max_runs=10, min_passes=3)
    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(TEST_QPU_LATTICES)])
    def test_2q_gate_pauliz_identity_tensor(self, device, qvm, compiler):
        """Test that the PauliZ tensor Identity observable works correctly.

        As the results coming from the qvm are stochastic, a constraint of 3 out of 5 runs was added.
        """
        dev = qml.device("forest.qvm", device=device, shots=QVM_SHOTS)

        @qml.qnode(dev)
        def circuit():
            qml.RY(np.pi / 2, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.Identity(1))

        assert np.allclose(circuit(), 0.0, atol=2e-2)

    @flaky(max_runs=10, min_passes=3)
    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(TEST_QPU_LATTICES)])
    def test_2q_gate_pauliz_pauliz_tensor(self, device, qvm, compiler):
        """Test that the PauliZ tensor PauliZ observable works correctly.

        As the results coming from the qvm are stochastic, a constraint of 3 out of 5 runs was added.
        """
        dev = qml.device("forest.qvm", device=device, shots=QVM_SHOTS)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert np.allclose(circuit(), 1.0, atol=2e-2)

    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(TEST_QPU_LATTICES)])
    def test_compiled_program_was_stored(self, qvm, device):
        """Test that QVM device stores the compiled program correctly"""
        dev = qml.device("forest.qvm", device=device, timeout=100)

        assert len(dev._compiled_program_dict.items()) == 0

        def circuit(params, wires):
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])

        obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
        obs_list = obs * 6

        qnodes = qml.map(circuit, obs_list, dev)

        qnodes([])
        assert dev.circuit_hash in dev._compiled_program_dict
        assert len(dev._compiled_program_dict.items()) == 1

    @pytest.mark.parametrize(
        "statements",
        [
            [True, True, True, True, True, True],
            [True, False, True, False, True, False],
            [True, False, False, False, True, False],
            [False, False, False, False, False, True],
            [True, False, False, False, False, False],
            [False, False, False, False, False, False],
        ],
    )
    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(TEST_QPU_LATTICES)])
    def test_compiled_program_was_stored_mutable_qnode_with_if_statement(
        self, qvm, device, statements
    ):
        """Test that QVM device stores the compiled program when the QNode is mutated correctly"""
        dev = qml.device("forest.qvm", device=device, timeout=100)

        assert len(dev._compiled_program_dict.items()) == 0

        def circuit(params, wires, statement=None):
            if statement:
                qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])

        obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
        obs_list = obs * 6

        qnodes = qml.map(circuit, obs_list, dev)

        for idx, stmt in enumerate(statements):
            qnodes[idx]([], statement=stmt)
            assert dev.circuit_hash in dev._compiled_program_dict

        # Using that True evaluates to 1
        number_of_true = sum(statements)

        # Checks if all elements in the list were either ``True`` or ``False``
        # In such a case we have compiled only one program
        length = 1 if (number_of_true == 6 or number_of_true == 0) else 2
        assert len(dev._compiled_program_dict.items()) == length

    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(TEST_QPU_LATTICES)])
    def test_compiled_program_was_stored_mutable_qnode_with_loop(self, qvm, device):
        """Test that QVM device stores the compiled program when the QNode is
        mutated correctly"""
        dev = qml.device("forest.qvm", device=device, timeout=80)

        assert len(dev._compiled_program_dict.items()) == 0

        def circuit(params, wires, rounds=1):
            for i in range(rounds):
                qml.Hadamard(0)
                qml.CNOT(wires=[0, 1])

        obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
        obs_list = obs * 6

        qnodes = qml.map(circuit, obs_list, dev)

        for idx, qnode in enumerate(qnodes):
            qnode([], rounds=idx)
            assert dev.circuit_hash in dev._compiled_program_dict

        assert len(dev._compiled_program_dict.items()) == len(qnodes)

    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(TEST_QPU_LATTICES)])
    def test_compiled_program_was_used(self, qvm, device, monkeypatch):
        """Test that QVM device used the compiled program correctly, after it was stored"""
        dev = qml.device("forest.qvm", device=device, timeout=100)

        number_of_qnodes = 6
        obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
        obs_list = obs * number_of_qnodes

        qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=dev.num_wires)
        params = np.random.random(size=shape)

        # For the first evaluation, use the real compile method
        qnodes[0](params)

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QuantumComputer, "compile", lambda self, prog: call_history.append(prog))

            for i in range(1, number_of_qnodes):
                qnodes[i](params)

        # Then use the mocked one to see if it was called

        results = qnodes(params)

        assert len(call_history) == 0
        assert dev.circuit_hash in dev._compiled_program_dict
        assert len(dev._compiled_program_dict.items()) == 1

    @flaky(max_runs=10, min_passes=1)
    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(TEST_QPU_LATTICES)])
    def test_compiled_program_was_correct_compared_with_default_qubit(self, qvm, device, tol):
        """Test that QVM device stores the compiled program correctly by comparing it with default.qubit.

        As the results coming from the qvm are stochastic, a constraint of 1 out of 5 runs was added.
        """
        number_of_qnodes = 6
        obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
        obs_list = obs * number_of_qnodes

        dev = qml.device("forest.qvm", device=device, timeout=100)

        shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=dev.num_wires)
        params = np.random.random(size=shape)

        qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev)

        results = qnodes(params)

        dev2 = qml.device("default.qubit", wires=dev.num_wires)
        qnodes2 = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev2)

        results2 = qnodes2(params)

        assert np.allclose(results, results2, atol=6e-02, rtol=0)
        assert dev.circuit_hash in dev._compiled_program_dict
        assert len(dev._compiled_program_dict.items()) == 1

    @flaky(max_runs=10, min_passes=3)
    @pytest.mark.parametrize("device", ["2q-qvm", np.random.choice(TEST_QPU_LATTICES)])
    def test_2q_gate_pauliz_pauliz_tensor_parametric_compilation_off(self, device, qvm, compiler):
        """Test that the PauliZ tensor PauliZ observable works correctly, when parametric compilation
        was turned off.

        As the results coming from the qvm are stochastic, a constraint of 3 out of 5 runs was added.
        """

        dev = qml.device("forest.qvm", device=device, shots=QVM_SHOTS, parametric_compilation=False)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert np.allclose(circuit(), 1.0, atol=2e-2)
