"""
Unit tests for the QPU device.
"""
import logging
from pyquil.experiment import SymmetrizationLevel

import pytest
import pyquil
import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import Tensor
from pennylane.wires import Wires
import pennylane_forest as plf
from conftest import BaseTest

from flaky import flaky

log = logging.getLogger(__name__)

TEST_QPU_LATTICES = ["4q-qvm"]


class TestQPUIntegration(BaseTest):
    """Test the wavefunction simulator works correctly from the PennyLane frontend."""

    # pylint: disable=no-self-use

    def test_load_qpu_device(self):
        """Test that the QPU device loads correctly"""
        device = TEST_QPU_LATTICES[0]
        dev = qml.device("forest.qpu", device=device, load_qc=False)
        qc = pyquil.get_qc(device)
        num_wires = len(qc.qubits())
        self.assertEqual(dev.num_wires, num_wires)
        self.assertEqual(dev.shots, 1000)
        self.assertEqual(dev.short_name, "forest.qpu")

    def test_load_virtual_qpu_device(self):
        """Test that the QPU simulators load correctly"""
        device = np.random.choice(TEST_QPU_LATTICES)
        qml.device("forest.qpu", device=device, load_qc=False)

    def test_qpu_args(self):
        """Test that the QPU plugin requires correct arguments"""
        device = np.random.choice(TEST_QPU_LATTICES)

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device("forest.qpu")

        with pytest.raises(ValueError, match="Number of shots must be a positive integer"):
            qml.device("forest.qpu", device=device, shots=0)

        with pytest.raises(ValueError, match="Readout error cannot be set on the physical QPU"):
            qml.device("forest.qpu", device=device, load_qc=True, readout_error=[0.9, 0.75])

        dev_no_wires = qml.device("forest.qpu", device=device, shots=5, load_qc=False)
        assert dev_no_wires.wires == Wires(range(4))

        with pytest.raises(ValueError, match="Device has a fixed number of"):
            qml.device("forest.qpu", device=device, shots=5, wires=100, load_qc=False)

        dev_iterable_wires = qml.device("forest.qpu", device=device, shots=5, wires=range(4), load_qc=False)
        assert dev_iterable_wires.wires == Wires(range(4))

        with pytest.raises(ValueError, match="Device has a fixed number of"):
            qml.device("forest.qpu", device=device, shots=5, wires=range(100), load_qc=False)

    @flaky(max_runs=10, min_passes=3)
    @pytest.mark.parametrize(
        "obs", [qml.PauliX(0), qml.PauliZ(0), qml.PauliY(0), qml.Hadamard(0), qml.Identity(0)]
    )
    def test_tensor_expval_parametric_compilation(self, obs):
        """Test the QPU expval method for Tensor observables made up of a single observable when parametric compilation is
        turned on.

        As the results coming from the qvm are stochastic, a constraint of 3 out of 10 runs was added.
        """
        device = np.random.choice(TEST_QPU_LATTICES)
        p = np.pi / 8
        dev = qml.device(
            "forest.qpu",
            device=device,
            shots=10000,
            load_qc=False,
            parametric_compilation=True,
        )
        dev_1 = qml.device(
            "forest.qpu",
            device=device,
            shots=10000,
            load_qc=False,
            parametric_compilation=True,
        )

        def template(param):
            qml.BasisState(np.array([0, 0, 1, 1]), wires=list(range(4)))
            qml.RY(param, wires=[2])
            qml.CNOT(wires=[2, 3])

        @qml.qnode(dev)
        def circuit_tensor(param):
            template(param)
            return qml.expval(Tensor(obs))

        @qml.qnode(dev_1)
        def circuit_obs(param):
            template(param)
            return qml.expval(obs)

        res = circuit_tensor(p)
        exp = circuit_obs(p)

        assert np.allclose(res, exp, atol=2e-2)

    @flaky(max_runs=10, min_passes=3)
    @pytest.mark.parametrize(
        "obs", [qml.PauliX(0), qml.PauliZ(0), qml.PauliY(0), qml.Hadamard(0), qml.Identity(0)]
    )
    def test_tensor_expval_operator_estimation(self, obs, shots):
        """Test the QPU expval method for Tensor observables made up of a single observable when parametric compilation is
        turned off allowing operator estimation.

        As the results coming from the qvm are stochastic, a constraint of 3 out of 10 runs was added.
        """
        device = np.random.choice(TEST_QPU_LATTICES)
        p = np.pi / 7
        dev = qml.device(
            "forest.qpu",
            device=device,
            shots=shots,
            load_qc=False,
            # Disabling this for now, conflicts with warning on qpu device
            # parametric_compilation=False,
        )
        dev_1 = qml.device(
            "forest.qpu",
            device=device,
            shots=shots,
            load_qc=False,
            # Disabling this for now, conflicts with warning on qpu device
            # parametric_compilation=False,
        )

        def template(param):
            qml.BasisState(np.array([0, 0, 1, 1]), wires=list(range(4)))
            qml.RY(param, wires=[2])
            qml.CNOT(wires=[2, 3])

        @qml.qnode(dev)
        def circuit_tensor(param):
            template(param)
            return qml.expval(Tensor(obs))

        @qml.qnode(dev_1)
        def circuit_obs(param):
            template(param)
            return qml.expval(obs)

        res = circuit_tensor(p)
        exp = circuit_obs(p)

        assert np.allclose(res, exp, atol=2e-2)


class TestQPUBasic(BaseTest):
    """Unit tests for the QPU (as a QVM)."""

    # pylint: disable=protected-access

    def test_warnings_raised_parametric_compilation_and_operator_estimation(self):
        """Test that a warning is raised if parameter compilation and operator estimation are both turned on."""
        device = np.random.choice(TEST_QPU_LATTICES)
        with pytest.warns(Warning, match="Operator estimation is being turned off."):
            dev = qml.device(
                "forest.qpu",
                device=device,
                shots=1000,
                load_qc=False,
                parametric_compilation=True,
            )

    def test_no_readout_correction(self, shots):
        """Test the QPU plugin with no readout correction"""
        device = np.random.choice(TEST_QPU_LATTICES)
        dev_qpu = qml.device(
            "forest.qpu",
            device=device,
            load_qc=False,
            readout_error=[0.9, 0.75],
            symmetrize_readout=SymmetrizationLevel.NONE,
            calibrate_readout=None,
            parametric_compilation=False,
            shots=shots
        )
        qubit = 0  # just run program on the first qubit

        @qml.qnode(dev_qpu)
        def circuit_Xpl():
            qml.RY(np.pi / 2, wires=qubit)
            return qml.expval(qml.PauliX(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Xmi():
            qml.RY(-np.pi / 2, wires=qubit)
            return qml.expval(qml.PauliX(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Ypl():
            qml.RX(-np.pi / 2, wires=qubit)
            return qml.expval(qml.PauliY(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Ymi():
            qml.RX(np.pi / 2, wires=qubit)
            return qml.expval(qml.PauliY(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Zpl():
            qml.RX(0.0, wires=qubit)
            return qml.expval(qml.PauliZ(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Zmi():
            qml.RX(np.pi, wires=qubit)
            return qml.expval(qml.PauliZ(qubit))

        num_expts = 10
        results_unavged = np.zeros((num_expts, 6))

        for i in range(num_expts):
            results_unavged[i, :] = [
                circuit_Xpl(),
                circuit_Ypl(),
                circuit_Zpl(),
                circuit_Xmi(),
                circuit_Ymi(),
                circuit_Zmi(),
            ]

        results = np.mean(results_unavged, axis=0)

        assert np.allclose(results[:3], 0.8, atol=2e-2)
        assert np.allclose(results[3:], -0.5, atol=2e-2)

    def test_readout_correction(self):
        """Test the QPU plugin with readout correction"""
        device = np.random.choice(TEST_QPU_LATTICES)
        dev_qpu = qml.device(
            "forest.qpu",
            device=device,
            load_qc=False,
            readout_error=[0.9, 0.75],
            symmetrize_readout=SymmetrizationLevel.EXHAUSTIVE,
            calibrate_readout="plus-eig",
            timeout=100,
        )
        qubit = 0  # just run program on the first qubit

        @qml.qnode(dev_qpu)
        def circuit_Xpl():
            qml.RY(np.pi / 2, wires=qubit)
            return qml.expval(qml.PauliX(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Xmi():
            qml.RY(-np.pi / 2, wires=qubit)
            return qml.expval(qml.PauliX(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Ypl():
            qml.RX(-np.pi / 2, wires=qubit)
            return qml.expval(qml.PauliY(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Ymi():
            qml.RX(np.pi / 2, wires=qubit)
            return qml.expval(qml.PauliY(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Zpl():
            qml.RX(0.0, wires=qubit)
            return qml.expval(qml.PauliZ(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Zmi():
            qml.RX(np.pi, wires=qubit)
            return qml.expval(qml.PauliZ(qubit))

        num_expts = 10
        results_unavged = np.zeros((num_expts, 6))

        for i in range(num_expts):
            results_unavged[i, :] = [
                circuit_Xpl(),
                circuit_Ypl(),
                circuit_Zpl(),
                circuit_Xmi(),
                circuit_Ymi(),
                circuit_Zmi(),
            ]

        results = np.mean(results_unavged, axis=0)

        assert np.allclose(results[:3], 1.0, atol=2e-2)
        assert np.allclose(results[3:], -1.0, atol=2e-2)

    @flaky(max_runs=10, min_passes=3)
    def test_multi_qub_no_readout_errors(self):
        """Test the QPU plugin with no readout errors or correction"""
        device = np.random.choice(TEST_QPU_LATTICES)
        dev_qpu = qml.device(
            "forest.qpu",
            device=device,
            load_qc=False,
            symmetrize_readout=SymmetrizationLevel.NONE,
            calibrate_readout=None,
        )

        @qml.qnode(dev_qpu)
        def circuit():
            qml.RY(np.pi / 2, wires=0)
            qml.RY(np.pi / 3, wires=1)
            return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

        num_expts = 50
        result = 0.0
        for _ in range(num_expts):
            result += circuit()
        result /= num_expts

        assert np.isclose(result, 0.5, atol=2e-2)

    @flaky(max_runs=10, min_passes=3)
    def test_multi_qub_readout_errors(self):
        """Test the QPU plugin with readout errors"""
        device = np.random.choice(TEST_QPU_LATTICES)
        dev_qpu = qml.device(
            "forest.qpu",
            device=device,
            load_qc=False,
            shots=10_000,
            readout_error=[0.9, 0.75],
            symmetrize_readout=SymmetrizationLevel.NONE,
            calibrate_readout=None,
            parametric_compilation=False
        )

        @qml.qnode(dev_qpu)
        def circuit():
            qml.RY(np.pi / 2, wires=0)
            qml.RY(np.pi / 3, wires=1)
            return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

        result = circuit()

        assert np.isclose(result, 0.38, atol=2e-2)

    @flaky(max_runs=10, min_passes=3)
    def test_multi_qub_readout_correction(self):
        """Test the QPU plugin with readout errors and correction"""
        device = np.random.choice(TEST_QPU_LATTICES)
        dev_qpu = qml.device(
            "forest.qpu",
            device=device,
            load_qc=False,
            shots=10_000,
            readout_error=[0.9, 0.75],
            symmetrize_readout=SymmetrizationLevel.EXHAUSTIVE,
            calibrate_readout='plus-eig',
            parametric_compilation=False
        )

        @qml.qnode(dev_qpu)
        def circuit():
            qml.RY(np.pi / 2, wires=0)
            qml.RY(np.pi / 3, wires=1)
            return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

        result = circuit()

        assert np.isclose(result, 0.5, atol=3e-2)

    @flaky(max_runs=10, min_passes=3)
    def test_2q_gate(self, shots):
        """Test that the two qubit gate with the PauliZ observable works correctly.

        As the results coming from the qvm are stochastic, a constraint of 3 out of 10 runs was added.
        """

        device = np.random.choice(TEST_QPU_LATTICES)
        dev_qpu = qml.device(
            "forest.qpu",
            device=device,
            load_qc=False,
            readout_error=[0.9, 0.75],
            symmetrize_readout=SymmetrizationLevel.EXHAUSTIVE,
            calibrate_readout="plus-eig",
            shots=shots,
        )

        @qml.qnode(dev_qpu)
        def circuit():
            qml.RY(np.pi / 2, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(), 0.0, atol=2e-2)

    @flaky(max_runs=10, min_passes=3)
    def test_2q_gate_pauliz_identity_tensor(self, shots):
        """Test that the PauliZ tensor Identity observable works correctly.

        As the results coming from the qvm are stochastic, a constraint of 3 out of 10 runs was added.
        """
        device = np.random.choice(TEST_QPU_LATTICES)
        dev_qpu = qml.device(
            "forest.qpu",
            device=device,
            load_qc=False,
            readout_error=[0.9, 0.75],
            symmetrize_readout=SymmetrizationLevel.EXHAUSTIVE,
            calibrate_readout="plus-eig",
            shots=shots,
        )

        @qml.qnode(dev_qpu)
        def circuit():
            qml.RY(np.pi / 2, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.Identity(1))

        assert np.allclose(circuit(), 0.0, atol=2e-2)

    @flaky(max_runs=10, min_passes=3)
    @pytest.mark.parametrize("a", np.linspace(-0.5, 2, 6))
    def test_2q_gate_pauliz_pauliz_tensor(self, a, shots):
        """Test that the PauliZ tensor PauliZ observable works correctly.

        As the results coming from the qvm are stochastic, a constraint of 3 out of 10 runs was added.
        """
        device = np.random.choice(TEST_QPU_LATTICES)
        dev_qpu = qml.device(
            "forest.qpu",
            device=device,
            load_qc=False,
            readout_error=[0.9, 0.75],
            symmetrize_readout=SymmetrizationLevel.EXHAUSTIVE,
            calibrate_readout="plus-eig",
            shots=shots,
        )

        @qml.qnode(dev_qpu)
        def circuit(x):
            qml.RY(x, wires=[0])
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.Identity(1))

        assert np.allclose(circuit(a), np.cos(a), atol=2e-2)
        # Check that repeated calling of the QNode works correctly
        assert np.allclose(circuit(a), np.cos(a), atol=2e-2)

    @flaky(max_runs=10, min_passes=3)
    @pytest.mark.parametrize("a", np.linspace(-np.pi / 2, 0, 3))
    @pytest.mark.parametrize("b", np.linspace(0, np.pi / 2, 3))
    def test_2q_circuit_pauliz_pauliz_tensor(self, a, b, shots):
        """Test that the PauliZ tensor PauliZ observable works correctly, when parametric compilation
        is turned off.

        As the results coming from the qvm are stochastic, a constraint of 3 out of 10 runs was added.
        """

        device = np.random.choice(TEST_QPU_LATTICES)
        dev_qpu = qml.device(
            "forest.qpu",
            device=device,
            load_qc=False,
            readout_error=[0.9, 0.75],
            symmetrize_readout=SymmetrizationLevel.EXHAUSTIVE,
            calibrate_readout="plus-eig",
            shots=shots,
        )

        @qml.qnode(dev_qpu)
        def circuit(x, y):
            qml.RY(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        analytic_value = (
            np.cos(a / 2) ** 2 * np.cos(b / 2) ** 2
            + np.cos(b / 2) ** 2 * np.sin(a / 2) ** 2
            - np.cos(a / 2) ** 2 * np.sin(b / 2) ** 2
            - np.sin(a / 2) ** 2 * np.sin(b / 2) ** 2
        )

        assert np.allclose(circuit(a, b), analytic_value, atol=2e-2)
        # Check that repeated calling of the QNode works correctly
        assert np.allclose(circuit(a, b), analytic_value, atol=2e-2)

    @flaky(max_runs=10, min_passes=3)
    @pytest.mark.parametrize("a", np.linspace(-np.pi / 2, 0, 3))
    @pytest.mark.parametrize("b", np.linspace(0, np.pi / 2, 3))
    def test_2q_gate_pauliz_pauliz_tensor_parametric_compilation_off(self, a, b, shots):
        """Test that the PauliZ tensor PauliZ observable works correctly, when parametric compilation
        is turned off.

        As the results coming from the qvm are stochastic, a constraint of 3 out of 10 runs was added.
        """

        device = np.random.choice(TEST_QPU_LATTICES)
        dev_qpu = qml.device(
            "forest.qpu",
            device=device,
            load_qc=False,
            readout_error=[0.9, 0.75],
            symmetrize_readout=SymmetrizationLevel.EXHAUSTIVE,
            calibrate_readout="plus-eig",
            shots=shots,
            parametric_compilation=False,
        )

        @qml.qnode(dev_qpu)
        def circuit(x, y):
            qml.RY(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        analytic_value = (
            np.cos(a / 2) ** 2 * np.cos(b / 2) ** 2
            + np.cos(b / 2) ** 2 * np.sin(a / 2) ** 2
            - np.cos(a / 2) ** 2 * np.sin(b / 2) ** 2
            - np.sin(a / 2) ** 2 * np.sin(b / 2) ** 2
        )

        expt = np.mean([circuit(a, b) for _ in range(20)])
        theory = analytic_value

        assert np.allclose(expt, theory, atol=2e-2)

    def test_timeout_set_correctly(self, shots):
        """Test that the timeout attrbiute for the QuantumComputer stored by the QVMDevice
        is set correctly when passing a value as keyword argument"""
        device = np.random.choice(TEST_QPU_LATTICES)
        dev = plf.QVMDevice(device=device, shots=shots, compiler_timeout=100, execution_timeout=101)
        assert dev.qc.compiler._timeout == 100
        assert dev.qc.qam._qvm_client.timeout == 101

    def test_timeout_default(self, shots):
        """Test that the timeout attrbiute for the QuantumComputer stored by the QVMDevice
        is set to default when no specific value is being passed."""
        device = np.random.choice(TEST_QPU_LATTICES)
        dev = plf.QVMDevice(device=device, shots=shots)
        qc = pyquil.get_qc(device, as_qvm=True)

        # Check that the timeouts are equal (it has not been changed as a side effect of
        # instantiation
        assert dev.qc.compiler._timeout == qc.compiler._timeout
        assert dev.qc.qam._qvm_client.timeout == qc.qam._qvm_client.timeout
