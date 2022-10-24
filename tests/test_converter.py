import os
import textwrap

import numpy as np
import pennylane as qml
import pyquil
import pyquil.gates as g
import pytest
from pennylane.tape import OperationRecorder
from pennylane_forest.converter import *


class TestProgramConverter:
    """Test that PyQuil Program instances are properly converted."""

    @pytest.mark.parametrize(
        "pyquil_operation,expected_pl_operation",
        [
            (g.I(0), qml.Identity(wires=[0])),
            (g.H(0), qml.Hadamard(0)),
            (g.H(0).dagger(), qml.Hadamard(0).inv()),
            (g.H(0).dagger().dagger(), qml.Hadamard(0).inv().inv()),
            (g.S(0), qml.S(wires=[0])),
            (g.S(0).dagger(), qml.S(wires=[0]).inv()),
            (g.S(0).dagger().dagger(), qml.S(wires=[0]).inv().inv()),
            (g.T(0), qml.T(wires=[0])),
            (g.T(0).dagger(), qml.T(wires=[0]).inv()),
            (g.T(0).dagger().dagger(), qml.T(wires=[0]).inv().inv()),
            (g.X(0), qml.PauliX(0)),
            (g.X(0).dagger(), qml.PauliX(0).inv()),
            (g.X(0).dagger().dagger(), qml.PauliX(0).inv().inv()),
            (g.X(0).controlled(1), qml.CNOT(wires=[1, 0])),
            (g.X(0).controlled(1).dagger(), qml.CNOT(wires=[1, 0]).inv()),
            (g.X(0).controlled(1).dagger().dagger(), qml.CNOT(wires=[1, 0]).inv().inv()),
            (g.X(0).controlled(1).controlled(2), qml.Toffoli(wires=[2, 1, 0])),
            (g.X(0).controlled(1).controlled(2).dagger(), qml.Toffoli(wires=[2, 1, 0]).inv()),
            (
                g.X(0).controlled(1).controlled(2).dagger().dagger(),
                qml.Toffoli(wires=[2, 1, 0]).inv().inv(),
            ),
            (g.Y(0), qml.PauliY(0)),
            (g.Y(0).dagger(), qml.PauliY(0).inv()),
            (g.Y(0).dagger().dagger(), qml.PauliY(0).inv().inv()),
            (g.Z(0), qml.PauliZ(0)),
            (g.Z(0).dagger(), qml.PauliZ(0).inv()),
            (g.Z(0).dagger().dagger(), qml.PauliZ(0).inv().inv()),
            (g.Z(0).controlled(1), qml.CZ(wires=[1, 0])),
            (g.Z(0).controlled(1).dagger(), qml.CZ(wires=[1, 0]).inv()),
            (g.Z(0).controlled(1).dagger().dagger(), qml.CZ(wires=[1, 0]).inv().inv()),
            (g.CNOT(0, 1), qml.CNOT(wires=[0, 1])),
            (g.CNOT(0, 1).dagger(), qml.CNOT(wires=[0, 1]).inv()),
            (g.CNOT(0, 1).dagger().dagger(), qml.CNOT(wires=[0, 1]).inv().inv()),
            (g.CNOT(0, 1).controlled(2), qml.Toffoli(wires=[2, 0, 1])),
            (g.CNOT(0, 1).controlled(2).dagger(), qml.Toffoli(wires=[2, 0, 1]).inv()),
            (
                g.CNOT(0, 1).controlled(2).dagger().dagger(),
                qml.Toffoli(wires=[2, 0, 1]).inv().inv(),
            ),
            (g.SWAP(0, 1), qml.SWAP(wires=[0, 1])),
            (g.SWAP(0, 1).dagger(), qml.SWAP(wires=[0, 1]).inv()),
            (g.SWAP(0, 1).dagger().dagger(), qml.SWAP(wires=[0, 1]).inv().inv()),
            (g.SWAP(0, 1).controlled(2), qml.CSWAP(wires=[2, 0, 1])),
            (g.SWAP(0, 1).controlled(2).dagger(), qml.CSWAP(wires=[2, 0, 1]).inv()),
            (g.SWAP(0, 1).controlled(2).dagger().dagger(), qml.CSWAP(wires=[2, 0, 1]).inv().inv()),
            (g.ISWAP(0, 1), qml.ISWAP(wires=[0, 1])),
            (g.ISWAP(0, 1).dagger(), qml.ISWAP(wires=[0, 1]).inv()),
            (g.ISWAP(0, 1).dagger().dagger(), qml.ISWAP(wires=[0, 1]).inv().inv()),
            (g.PSWAP(0.3, 0, 1), qml.PSWAP(0.3, wires=[0, 1])),
            (g.PSWAP(0.3, 0, 1).dagger(), qml.PSWAP(0.3, wires=[0, 1]).inv()),
            (g.PSWAP(0.3, 0, 1).dagger().dagger(), qml.PSWAP(0.3, wires=[0, 1]).inv().inv()),
            (g.CZ(0, 1), qml.CZ(wires=[0, 1])),
            (g.CZ(0, 1).dagger(), qml.CZ(wires=[0, 1]).inv()),
            (g.CZ(0, 1).dagger().dagger(), qml.CZ(wires=[0, 1]).inv().inv()),
            (g.PHASE(0.3, 0), qml.PhaseShift(0.3, wires=[0])),
            (g.PHASE(0.3, 0).dagger(), qml.PhaseShift(0.3, wires=[0]).inv()),
            (g.PHASE(0.3, 0).dagger().dagger(), qml.PhaseShift(0.3, wires=[0]).inv().inv()),
            (g.PHASE(0.3, 0).controlled(1), plf.ops.CPHASE(0.3, 3, wires=[1, 0])),
            (g.PHASE(0.3, 0).controlled(1).dagger(), plf.ops.CPHASE(0.3, 3, wires=[1, 0]).inv()),
            (
                g.PHASE(0.3, 0).controlled(1).dagger().dagger(),
                plf.ops.CPHASE(0.3, 3, wires=[1, 0]).inv().inv(),
            ),
            (g.RX(0.3, 0), qml.RX(0.3, wires=[0])),
            (g.RX(0.3, 0).dagger(), qml.RX(0.3, wires=[0]).inv()),
            (g.RX(0.3, 0).dagger().dagger(), qml.RX(0.3, wires=[0]).inv().inv()),
            (g.RX(0.3, 0).controlled(1), qml.CRX(0.3, wires=[1, 0])),
            (g.RX(0.3, 0).controlled(1).dagger(), qml.CRX(0.3, wires=[1, 0]).inv()),
            (g.RX(0.3, 0).controlled(1).dagger().dagger(), qml.CRX(0.3, wires=[1, 0]).inv().inv()),
            (g.RY(0.3, 0), qml.RY(0.3, wires=[0])),
            (g.RY(0.3, 0).dagger(), qml.RY(0.3, wires=[0]).inv()),
            (g.RY(0.3, 0).dagger().dagger(), qml.RY(0.3, wires=[0]).inv().inv()),
            (g.RY(0.3, 0).controlled(1), qml.CRY(0.3, wires=[1, 0])),
            (g.RY(0.3, 0).controlled(1).dagger(), qml.CRY(0.3, wires=[1, 0]).inv()),
            (g.RY(0.3, 0).controlled(1).dagger().dagger(), qml.CRY(0.3, wires=[1, 0]).inv().inv()),
            (g.RZ(0.3, 0), qml.RZ(0.3, wires=[0])),
            (g.RZ(0.3, 0).dagger(), qml.RZ(0.3, wires=[0]).inv()),
            (g.RZ(0.3, 0).dagger().dagger(), qml.RZ(0.3, wires=[0]).inv().inv()),
            (g.RZ(0.3, 0).controlled(1), qml.CRZ(0.3, wires=[1, 0])),
            (g.RZ(0.3, 0).controlled(1).dagger(), qml.CRZ(0.3, wires=[1, 0]).inv()),
            (g.RZ(0.3, 0).controlled(1).dagger().dagger(), qml.CRZ(0.3, wires=[1, 0]).inv().inv()),
            (g.CPHASE(0.3, 0, 1), plf.ops.CPHASE(0.3, 3, wires=[0, 1])),
            (g.CPHASE(0.3, 0, 1).dagger(), plf.ops.CPHASE(0.3, 3, wires=[0, 1]).inv()),
            (
                g.CPHASE(0.3, 0, 1).dagger().dagger(),
                plf.ops.CPHASE(0.3, 3, wires=[0, 1]).inv().inv(),
            ),
            (g.CPHASE00(0.3, 0, 1), plf.ops.CPHASE(0.3, 0, wires=[0, 1])),
            (g.CPHASE00(0.3, 0, 1).dagger(), plf.ops.CPHASE(0.3, 0, wires=[0, 1]).inv()),
            (
                g.CPHASE00(0.3, 0, 1).dagger().dagger(),
                plf.ops.CPHASE(0.3, 0, wires=[0, 1]).inv().inv(),
            ),
            (g.CPHASE01(0.3, 0, 1), plf.ops.CPHASE(0.3, 1, wires=[0, 1])),
            (g.CPHASE01(0.3, 0, 1).dagger(), plf.ops.CPHASE(0.3, 1, wires=[0, 1]).inv()),
            (
                g.CPHASE01(0.3, 0, 1).dagger().dagger(),
                plf.ops.CPHASE(0.3, 1, wires=[0, 1]).inv().inv(),
            ),
            (g.CPHASE10(0.3, 0, 1), plf.ops.CPHASE(0.3, 2, wires=[0, 1])),
            (g.CPHASE10(0.3, 0, 1).dagger(), plf.ops.CPHASE(0.3, 2, wires=[0, 1]).inv()),
            (
                g.CPHASE10(0.3, 0, 1).dagger().dagger(),
                plf.ops.CPHASE(0.3, 2, wires=[0, 1]).inv().inv(),
            ),
            (g.CSWAP(0, 1, 2), qml.CSWAP(wires=[0, 1, 2])),
            (g.CSWAP(0, 1, 2).dagger(), qml.CSWAP(wires=[0, 1, 2]).inv()),
            (g.CSWAP(0, 1, 2).dagger().dagger(), qml.CSWAP(wires=[0, 1, 2]).inv().inv()),
            (g.CCNOT(0, 1, 2), qml.Toffoli(wires=[0, 1, 2])),
            (g.CCNOT(0, 1, 2).dagger(), qml.Toffoli(wires=[0, 1, 2]).inv()),
            (g.CCNOT(0, 1, 2).dagger().dagger(), qml.Toffoli(wires=[0, 1, 2]).inv().inv()),
        ],
    )
    def test_convert_operation(self, pyquil_operation, expected_pl_operation):
        """Test that single pyquil gates are properly converted."""
        program = pyquil.Program()

        program += pyquil_operation

        with OperationRecorder() as rec:
            loader = load_program(program)
            loader(wires=range(len(loader.defined_qubits)))

        assert rec.queue[0].name == expected_pl_operation.name
        assert rec.queue[0].wires == expected_pl_operation.wires
        assert rec.queue[0].data == expected_pl_operation.data

    def test_convert_simple_program(self):
        """Test that a simple program is properly converted."""
        program = pyquil.Program()

        program += g.H(0)
        program += g.RZ(0.34, 1)
        program += g.CNOT(0, 3)
        program += g.H(2)
        program += g.H(7)
        program += g.X(7)
        program += g.Y(1)
        program += g.RZ(0.34, 1)

        with OperationRecorder() as rec:
            load_program(program)(wires=range(5))

        # The wires should be assigned as
        # 0  1  2  3  7
        # 0  1  2  3  4

        expected_queue = [
            qml.Hadamard(0),
            qml.RZ(0.34, wires=[1]),
            qml.CNOT(wires=[0, 3]),
            qml.Hadamard(2),
            qml.Hadamard(4),
            qml.PauliX(4),
            qml.PauliY(1),
            qml.RZ(0.34, wires=[1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_convert_simple_program_with_parameters(self):
        """Test that a simple program with parameters is properly converted."""
        program = pyquil.Program()

        alpha = program.declare("alpha", "REAL")
        beta = program.declare("beta", "REAL")
        gamma = program.declare("gamma", "REAL")

        program += g.H(0)
        program += g.CNOT(0, 1)
        program += g.RX(alpha, 1)
        program += g.RZ(beta, 1)
        program += g.RX(gamma, 1)
        program += g.CNOT(0, 1)
        program += g.H(0)

        a, b, c = 0.1, 0.2, 0.3

        parameter_map = {"alpha": a, "beta": b, "gamma": c}

        with OperationRecorder() as rec:
            load_program(program)(wires=range(2), parameter_map=parameter_map)

        expected_queue = [
            qml.Hadamard(0),
            qml.CNOT(wires=[0, 1]),
            qml.RX(0.1, wires=[1]),
            qml.RZ(0.2, wires=[1]),
            qml.RX(0.3, wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(0),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_parameter_not_given_error(self):
        """Test that the correct error is raised if a parameter is not given."""
        program = pyquil.Program()

        alpha = program.declare("alpha", "REAL")
        beta = program.declare("beta", "REAL")

        program += g.H(0)
        program += g.CNOT(0, 1)
        program += g.RX(alpha, 1)
        program += g.RZ(beta, 1)

        a = 0.1

        parameter_map = {"alpha": a}

        with pytest.raises(
            qml.DeviceError,
            match="The PyQuil program defines a variable .* that is not present in the given variable map",
        ):
            load_program(program)(wires=range(2), parameter_map=parameter_map)

    def test_convert_simple_program_with_parameters_mixed_keys(self):
        """Test that a parametrized program is properly converted when
        the variable map contains mixed key types."""
        program = pyquil.Program()

        alpha = program.declare("alpha", "REAL")
        beta = program.declare("beta", "REAL")
        gamma = program.declare("gamma", "REAL")
        delta = program.declare("delta", "REAL")

        program += g.H(0)
        program += g.CNOT(0, 1)
        program += g.RX(alpha, 1)
        program += g.RZ(beta, 1)
        program += g.RX(gamma, 1)
        program += g.CNOT(0, 1)
        program += g.RZ(delta, 0)
        program += g.H(0)

        a, b, c, d = 0.1, 0.2, 0.3, 0.4

        parameter_map = {"alpha": a, beta: b, gamma: c, "delta": d}

        with OperationRecorder() as rec:
            load_program(program)(wires=range(2), parameter_map=parameter_map)

        expected_queue = [
            qml.Hadamard(0),
            qml.CNOT(wires=[0, 1]),
            qml.RX(0.1, wires=[1]),
            qml.RZ(0.2, wires=[1]),
            qml.RX(0.3, wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(0.4, wires=[0]),
            qml.Hadamard(0),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_convert_simple_program_wire_assignment(self):
        """Test that the assignment of qubits to wires works as expected."""
        program = pyquil.Program()

        program += g.H(0)
        program += g.RZ(0.34, 1)
        program += g.CNOT(0, 3)
        program += g.H(2)
        program += g.H(7)
        program += g.X(7)
        program += g.Y(1)
        program += g.RZ(0.34, 1)

        with OperationRecorder() as rec:
            load_program(program)(wires=[3, 6, 4, 9, 1])

        # The wires should be assigned as
        # 0  1  2  3  7
        # 3  6  4  9  1

        expected_queue = [
            qml.Hadamard(3),
            qml.RZ(0.34, wires=[6]),
            qml.CNOT(wires=[3, 9]),
            qml.Hadamard(4),
            qml.Hadamard(1),
            qml.PauliX(1),
            qml.PauliY(6),
            qml.RZ(0.34, wires=[6]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    @pytest.mark.parametrize("wires", [[0, 1, 2, 3], [4, 5]])
    def test_convert_wire_error(self, wires):
        """Test that the conversion raises an error if the given number
        of wires doesn't match the number of qubits in the Program."""
        program = pyquil.Program()

        program += g.H(0)
        program += g.H(1)
        program += g.H(2)

        with pytest.raises(
            qml.DeviceError,
            match="The number of given wires does not match the number of qubits in the PyQuil Program",
        ):
            load_program(program)(wires=wires)

    def test_convert_program_with_inverses(self):
        """Test that a program with inverses is properly converted."""
        program = pyquil.Program()

        program += g.H(0)
        program += g.RZ(0.34, 1).dagger()
        program += g.CNOT(0, 3).dagger()
        program += g.H(2)
        program += g.H(7).dagger().dagger()
        program += g.X(7).dagger()
        program += g.X(7)
        program += g.Y(1)
        program += g.RZ(0.34, 1)

        with OperationRecorder() as rec:
            load_program(program)(wires=range(5))

        expected_queue = [
            qml.Hadamard(0),
            qml.RZ(0.34, wires=[1]).inv(),
            qml.CNOT(wires=[0, 3]).inv(),
            qml.Hadamard(2),
            qml.Hadamard(4),
            qml.PauliX(4).inv(),
            qml.PauliX(4),
            qml.PauliY(1),
            qml.RZ(0.34, wires=[1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_convert_program_with_controlled_operations(self):
        """Test that a program with controlled operations is properly converted."""
        program = pyquil.Program()

        program += g.RZ(0.34, 1)
        program += g.RY(0.2, 3).controlled(2)
        program += g.RX(0.4, 2).controlled(0)
        program += g.CNOT(1, 4)
        program += g.CNOT(1, 6).controlled(3)
        program += g.X(3).controlled(4).controlled(1)

        with OperationRecorder() as rec:
            load_program(program)(wires=range(6))

        expected_queue = [
            qml.RZ(0.34, wires=[1]),
            qml.CRY(0.2, wires=[2, 3]),
            qml.CRX(0.4, wires=[0, 2]),
            qml.CNOT(wires=[1, 4]),
            qml.Toffoli(wires=[3, 1, 5]),
            qml.Toffoli(wires=[1, 4, 3]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_convert_program_with_controlled_operations_not_in_pl_core(self, tol):
        """Test that a program with controlled operations out of scope of PL core/PLF
        is properly converted, i.e. the operations are replaced with controlled operations."""
        program = pyquil.Program()

        CS_matrix = np.eye(4, dtype=complex)
        CS_matrix[3, 3] = 1j

        CCT_matrix = np.eye(8, dtype=complex)
        CCT_matrix[7, 7] = np.exp(1j * np.pi / 4)

        program += g.CNOT(0, 1)
        program += g.S(0).controlled(1)
        program += g.S(1).controlled(0)
        program += g.T(0).controlled(1).controlled(2)
        program += g.T(1).controlled(0).controlled(2)
        program += g.T(2).controlled(1).controlled(0)

        with OperationRecorder() as rec:
            load_program(program)(wires=range(3))

        expected_queue = [
            qml.CNOT(wires=[0, 1]),
            qml.QubitUnitary(CS_matrix, wires=[1, 0]),
            qml.QubitUnitary(CS_matrix, wires=[0, 1]),
            qml.QubitUnitary(CCT_matrix, wires=[2, 1, 0]),
            qml.QubitUnitary(CCT_matrix, wires=[2, 0, 1]),
            qml.QubitUnitary(CCT_matrix, wires=[0, 1, 2]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert np.allclose(converted.data, expected.data, atol=tol, rtol=0)

    def test_convert_program_with_controlled_dagger_operations(self):
        """Test that a program that combines controlled and daggered operations
        is properly converted."""
        program = pyquil.Program()

        program += g.CNOT(0, 1).controlled(2)
        program += g.CNOT(0, 1).dagger().controlled(2)
        program += g.CNOT(0, 1).controlled(2).dagger()
        program += g.CNOT(0, 1).dagger().controlled(2).dagger()
        program += g.RX(0.3, 3).controlled(4)
        program += g.RX(0.2, 3).controlled(4).dagger()
        program += g.RX(0.3, 3).dagger().controlled(4)
        program += g.RX(0.2, 3).dagger().controlled(4).dagger()
        program += g.X(2).dagger().controlled(4).controlled(1).dagger()
        program += g.X(0).dagger().controlled(4).controlled(1)
        program += g.X(0).dagger().controlled(4).dagger().dagger().controlled(1).dagger()

        with OperationRecorder() as rec:
            load_program(program)(wires=range(5))

        expected_queue = [
            qml.Toffoli(wires=[2, 0, 1]),
            qml.Toffoli(wires=[2, 0, 1]).inv(),
            qml.Toffoli(wires=[2, 0, 1]).inv(),
            qml.Toffoli(wires=[2, 0, 1]),
            qml.CRX(0.3, wires=[4, 3]),
            qml.CRX(0.2, wires=[4, 3]).inv(),
            qml.CRX(0.3, wires=[4, 3]).inv(),
            qml.CRX(0.2, wires=[4, 3]),
            qml.Toffoli(wires=[1, 4, 2]),
            qml.Toffoli(wires=[1, 4, 0]).inv(),
            qml.Toffoli(wires=[1, 4, 0]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_convert_program_with_defgates(self):
        """Test that a program that defines its own gates is properly converted."""
        program = pyquil.Program()

        sqrt_x = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])

        sqrt_x_t2 = np.kron(sqrt_x, sqrt_x)
        sqrt_x_t3 = np.kron(sqrt_x, sqrt_x_t2)

        sqrt_x_definition = pyquil.quil.DefGate("SQRT-X", sqrt_x)
        SQRT_X = sqrt_x_definition.get_constructor()
        sqrt_x_t2_definition = pyquil.quil.DefGate("SQRT-X-T2", sqrt_x_t2)
        SQRT_X_T2 = sqrt_x_t2_definition.get_constructor()
        sqrt_x_t3_definition = pyquil.quil.DefGate("SQRT-X-T3", sqrt_x_t3)
        SQRT_X_T3 = sqrt_x_t3_definition.get_constructor()

        program += sqrt_x_definition
        program += sqrt_x_t2_definition
        program += sqrt_x_t3_definition

        program += g.CNOT(0, 1)
        program += SQRT_X(0)
        program += SQRT_X_T2(1, 2)
        program += SQRT_X_T3(1, 0, 2)
        program += g.CNOT(0, 1)
        program += g.CNOT(1, 2)
        program += g.CNOT(2, 0)

        with OperationRecorder() as rec:
            load_program(program)(wires=range(3))

        expected_queue = [
            qml.CNOT(wires=[0, 1]),
            qml.QubitUnitary(sqrt_x, wires=[0]),
            qml.QubitUnitary(sqrt_x_t2, wires=[1, 2]),
            qml.QubitUnitary(sqrt_x_t3, wires=[1, 0, 2]),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 2]),
            qml.CNOT(wires=[2, 0]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_convert_program_with_controlled_defgates(self, tol):
        """Test that a program with controlled defined gates is properly
        converted."""
        program = pyquil.Program()

        sqrt_x = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])
        sqrt_x_t2 = np.kron(sqrt_x, sqrt_x)

        c_sqrt_x = np.eye(4, dtype=complex)
        c_sqrt_x[2:, 2:] = sqrt_x

        c_sqrt_x_t2 = np.eye(8, dtype=complex)
        c_sqrt_x_t2[4:, 4:] = sqrt_x_t2

        sqrt_x_definition = pyquil.quil.DefGate("SQRT-X", sqrt_x)
        SQRT_X = sqrt_x_definition.get_constructor()
        sqrt_x_t2_definition = pyquil.quil.DefGate("SQRT-X-T2", sqrt_x_t2)
        SQRT_X_T2 = sqrt_x_t2_definition.get_constructor()

        program += sqrt_x_definition
        program += sqrt_x_t2_definition

        program += g.CNOT(0, 1)
        program += SQRT_X(0).controlled(1)
        program += SQRT_X_T2(1, 2).controlled(0)
        program += g.X(0).controlled(1)
        program += g.RX(0.4, 0)

        with OperationRecorder() as rec:
            load_program(program)(wires=range(3))

        expected_queue = [
            qml.CNOT(wires=[0, 1]),
            qml.QubitUnitary(c_sqrt_x, wires=[1, 0]),
            qml.QubitUnitary(c_sqrt_x_t2, wires=[0, 1, 2]),
            qml.CNOT(wires=[1, 0]),
            qml.RX(0.4, wires=[0]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert np.allclose(converted.data, expected.data, atol=tol, rtol=0)

    def test_convert_program_with_defpermutationgates(self):
        """Test that a program with gates defined via DefPermutationGate is
        properly converted."""
        program = pyquil.Program()

        expected_matrix = np.eye(4)
        expected_matrix = expected_matrix[:, [1, 0, 3, 2]]

        x_plus_x_definition = pyquil.quil.DefPermutationGate("X+X", [1, 0, 3, 2])
        X_plus_X = x_plus_x_definition.get_constructor()

        program += x_plus_x_definition

        program += g.CNOT(0, 1)
        program += X_plus_X(0, 1)
        program += g.CNOT(0, 1)

        with OperationRecorder() as rec:
            load_program(program)(wires=range(2))

        expected_queue = [
            qml.CNOT(wires=[0, 1]),
            qml.QubitUnitary(expected_matrix, wires=[0, 1]),
            qml.CNOT(wires=[0, 1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert np.array_equal(converted.data, expected.data)

    def test_convert_program_with_controlled_defpermutationgates(self):
        """Test that a program that uses controlled permutation gates
        is properly converted."""
        program = pyquil.Program()

        expected_matrix = np.eye(4)
        expected_matrix = expected_matrix[:, [1, 0, 3, 2]]

        expected_controlled_matrix = np.eye(8)
        expected_controlled_matrix[4:, 4:] = expected_matrix

        x_plus_x_definition = pyquil.quil.DefPermutationGate("X+X", [1, 0, 3, 2])
        X_plus_X = x_plus_x_definition.get_constructor()

        program += x_plus_x_definition

        program += g.CNOT(0, 1)
        program += X_plus_X(0, 1).controlled(2)
        program += X_plus_X(1, 2).controlled(0)
        program += g.CNOT(0, 1)

        with OperationRecorder() as rec:
            load_program(program)(wires=range(3))

        expected_queue = [
            qml.CNOT(wires=[0, 1]),
            qml.QubitUnitary(expected_controlled_matrix, wires=[2, 0, 1]),
            qml.QubitUnitary(expected_controlled_matrix, wires=[0, 1, 2]),
            qml.CNOT(wires=[0, 1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert np.array_equal(converted.data, expected.data)

    def test_forked_gate_error(self):
        """Test that an error is raised if conversion of a
        forked gate is attempted."""
        program = pyquil.Program()

        program += g.CNOT(0, 1)
        program += g.RX(0.3, 1).forked(2, [0.5])
        program += g.CNOT(0, 1)

        with pytest.raises(
            qml.DeviceError,
            match="Forked gates can not be imported into PennyLane, as this functionality is not supported",
        ):
            load_program(program)(wires=range(3))


class TestQuilConverter:
    """Test that quil instances passed as string are properly converted."""

    def test_convert_simple_program(self):
        """Test that a simple program is properly converted."""
        quil_str = textwrap.dedent(
            """
            H 0
            RZ(0.34) 1
            CNOT 0 3
            H 2
            H 7
            X 7
            Y 1
            RZ(0.34) 1
        """
        )

        with OperationRecorder() as rec:
            load_quil(quil_str)(wires=range(5))

        # The wires should be assigned as
        # 0  1  2  3  7
        # 0  1  2  3  4

        expected_queue = [
            qml.Hadamard(0),
            qml.RZ(0.34, wires=[1]),
            qml.CNOT(wires=[0, 3]),
            qml.Hadamard(2),
            qml.Hadamard(4),
            qml.PauliX(4),
            qml.PauliY(1),
            qml.RZ(0.34, wires=[1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_convert_program_with_classical_control_flow(self):
        """Test that a program with classical control flow is properly converted."""
        quil_str = textwrap.dedent(
            """
            DECLARE flag_register BIT[1]
            MOVE flag_register 1
            LABEL @START1
            JUMP-UNLESS @END2 flag_register
            X 0
            H 0
            MEASURE 0 flag_register
            JUMP @START1
            LABEL @END2
        """
        )

        flag = 0.0

        with OperationRecorder() as rec:
            load_quil(quil_str)(wires=[0], parameter_map={"flag_register": flag})

        expected_queue = [qml.PauliX(0), qml.Hadamard(0)]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_convert_program_with_pragmas(self):
        """Test that a program with pragmas is properly converted."""
        quil_str = textwrap.dedent(
            """
            PRAGMA INITIAL_REWIRING "GREEDY"
            PRAGMA PRESERVE_BLOCK
            I 0
            I 1
            PRAGMA END_PRESERVE_BLOCK
            PRAGMA COMMUTING_BLOCKS
            PRAGMA BLOCK
            CNOT 0 1
            RZ(0.4) 1
            CNOT 0 1
            PRAGMA END_BLOCK
            PRAGMA BLOCK
            CNOT 1 2
            RZ(0.6) 2
            CNOT 1 2
            PRAGMA END_BLOCK
            PRAGMA END_COMMUTING_BLOCKS
            H 0
        """
        )

        with OperationRecorder() as rec:
            load_quil(quil_str)(wires=range(3))

        expected_queue = [
            qml.Identity(wires=[0]),
            qml.Identity(wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(0.4, wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 2]),
            qml.RZ(0.6, wires=[2]),
            qml.CNOT(wires=[1, 2]),
            qml.Hadamard(0),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_convert_complex_program(self):
        """Test that a more complicated program is properly converted."""
        quil_str = textwrap.dedent(
            """
            H 0
            DAGGER RZ(0.34) 1
            CNOT 0 3
            H 2
            CONTROLLED S 1 7
            DAGGER CONTROLLED X 3 7
            DAGGER DAGGER Y 1
            RZ(0.34) 1
        """
        )

        with OperationRecorder() as rec:
            load_quil(quil_str)(wires=range(5))

        # The wires should be assigned as
        # 0  1  2  3  7
        # 0  1  2  3  4

        CS_matrix = np.eye(4, dtype=complex)
        CS_matrix[3, 3] = 1j

        expected_queue = [
            qml.Hadamard(0),
            qml.RZ(0.34, wires=[1]).inv(),
            qml.CNOT(wires=[0, 3]),
            qml.Hadamard(2),
            qml.QubitUnitary(CS_matrix, wires=[1, 4]),
            qml.CNOT(wires=[3, 4]).inv(),
            qml.PauliY(1),
            qml.RZ(0.34, wires=[1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert np.array_equal(converted.data, expected.data)

    def test_convert_simple_program_with_parameters(self):
        """Test that a simple parametrized program is properly converted."""
        quil_str = textwrap.dedent(
            """
            DECLARE alpha REAL[1]
            DECLARE beta REAL[1]
            DECLARE gamma REAL[1]

            H 0
            CNOT 0 1
            RX(alpha) 1
            RZ(beta) 1
            RX(gamma) 1
            CNOT 0 1
            H 0
        """
        )

        a, b, c = 0.1, 0.2, 0.3

        parameter_map = {"alpha": a, "beta": b, "gamma": c}

        with OperationRecorder() as rec:
            load_quil(quil_str)(wires=range(2), parameter_map=parameter_map)

        expected_queue = [
            qml.Hadamard(0),
            qml.CNOT(wires=[0, 1]),
            qml.RX(0.1, wires=[1]),
            qml.RZ(0.2, wires=[1]),
            qml.RX(0.3, wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(0),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_convert_program_with_parameters_and_measurements(self):
        """Test that a program with parameters and measurements is properly converted."""
        quil_str = textwrap.dedent(
            """
            DECLARE alpha REAL[1]
            DECLARE beta REAL[1]
            DECLARE gamma REAL[1]
            DECLARE ro BIT[2]

            H 0
            CNOT 0 1
            RX(alpha) 1
            RZ(beta) 1
            RX(gamma) 1
            CNOT 0 1
            H 0

            MEASURE 0 ro[0]
            MEASURE 1 ro[1]
        """
        )

        a, b, c = 0.1, 0.2, 0.3

        parameter_map = {"alpha": a, "beta": b, "gamma": c}

        with OperationRecorder() as rec:
            load_quil(quil_str)(wires=range(2), parameter_map=parameter_map)

        expected_queue = [
            qml.Hadamard(0),
            qml.CNOT(wires=[0, 1]),
            qml.RX(0.1, wires=[1]),
            qml.RZ(0.2, wires=[1]),
            qml.RX(0.3, wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(0),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_convert_program_with_defgates(self):
        """Test that a program with defined gates is properly converted."""
        quil_str = textwrap.dedent(
            """
            DEFGATE SQRT-X:
                0.5+0.5i, 0.5-0.5i
                0.5-0.5i, 0.5+0.5i

            H 0
            CNOT 0 1
            SQRT-X 0
            SQRT-X 1
            H 1
            CONTROLLED SQRT-X 0 1
        """
        )

        sqrt_x = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])
        c_sqrt_x = np.eye(4, dtype=complex)
        c_sqrt_x[2:, 2:] = sqrt_x

        with OperationRecorder() as rec:
            load_quil(quil_str)(wires=range(2))

        expected_queue = [
            qml.Hadamard(0),
            qml.CNOT(wires=[0, 1]),
            qml.QubitUnitary(sqrt_x, wires=[0]),
            qml.QubitUnitary(sqrt_x, wires=[1]),
            qml.Hadamard(1),
            qml.QubitUnitary(c_sqrt_x, wires=[0, 1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert np.array_equal(converted.data, expected.data)


class TestQuilFileConverter:
    """Test that quil files are properly converted."""

    def test_convert_simple_program(self):
        """Test that a simple program in a quil file is properly converted."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        with OperationRecorder() as rec:
            load_quil_from_file(os.path.join(cur_dir, "simple_program.quil"))(wires=range(5))

        # The wires should be assigned as
        # 0  1  2  3  7
        # 0  1  2  3  4

        expected_queue = [
            qml.Hadamard(0),
            qml.RZ(0.34, wires=[1]),
            qml.CNOT(wires=[0, 3]),
            qml.Hadamard(2),
            qml.Hadamard(4),
            qml.PauliX(4),
            qml.PauliY(1),
            qml.RZ(0.34, wires=[1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data


class TestInspectionProperties:
    """Test that the inspection properties of ProgramLoader return the expected values."""

    def test_inspection(self):
        """Test the combined inspection properties."""
        quil_str = textwrap.dedent(
            """
            DECLARE alpha REAL[1]
            DECLARE beta REAL[1]
            DECLARE gamma REAL[1]

            DEFGATE SQRT-X:
                0.5+0.5i, 0.5-0.5i
                0.5-0.5i, 0.5+0.5i

            H 0
            DAGGER RZ(0.34) 1
            CNOT 0 3
            H 2
            RX(alpha) 1
            RZ(beta) 1
            CONTROLLED S 1 7
            DAGGER CONTROLLED X 3 7
            DAGGER DAGGER Y 1
            H 1
            CONTROLLED SQRT-X 0 1
            RZ(0.34) 1
        """
        )

        sqrt_x = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])

        loader = load_quil(quil_str)

        assert loader.declarations[0].name == "alpha"
        assert loader.declarations[1].name == "beta"
        assert loader.declarations[2].name == "gamma"

        assert loader.defined_variable_names[0] == "alpha"
        assert loader.defined_variable_names[1] == "beta"
        assert loader.defined_variable_names[2] == "gamma"

        assert loader.defined_gates[0].name == "SQRT-X"

        assert loader.defined_gate_names[0] == "SQRT-X"

        assert loader.defined_qubits == [0, 1, 2, 3, 7]

    def test_str(self):
        """Test that the string representation of ProgramLoader is correct."""
        quil_str = textwrap.dedent(
            """
            DECLARE alpha REAL[1]
            DECLARE beta REAL[1]
            DECLARE gamma REAL[1]
            DEFGATE SQRT-X:
                0.5+0.5i, 0.5-0.5i
                0.5-0.5i, 0.5+0.5i
            H 0
            DAGGER RZ(0.34) 1
            CNOT 0 3
            H 2
            RX(alpha) 1
            RZ(beta) 1
            CONTROLLED S 1 7
            DAGGER CONTROLLED X 3 7
            DAGGER DAGGER Y 1
            H 1
            CONTROLLED SQRT-X 0 1
            RZ(0.34) 1
        """
        )

        loader = load_quil(quil_str)

        assert str(loader) == "PennyLane Program Loader for PyQuil Program:\n" + str(loader.program)


class TestIntegration:
    """Test that the program loader integrates properly with PennyLane."""

    def test_load_program_via_entry_point(self):
        """Test that a pyquil Program instance can be loaded via the load entrypoint."""
        program = pyquil.Program()

        program += g.H(0)
        program += g.RZ(0.34, 1)
        program += g.CNOT(0, 3)
        program += g.H(2)
        program += g.H(7)
        program += g.X(7)
        program += g.Y(1)
        program += g.RZ(0.34, 1)

        with OperationRecorder() as rec:
            qml.load(program, format="pyquil_program")(wires=range(5))

        # The wires should be assigned as
        # 0  1  2  3  7
        # 0  1  2  3  4

        expected_queue = [
            qml.Hadamard(0),
            qml.RZ(0.34, wires=[1]),
            qml.CNOT(wires=[0, 3]),
            qml.Hadamard(2),
            qml.Hadamard(4),
            qml.PauliX(4),
            qml.PauliY(1),
            qml.RZ(0.34, wires=[1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_load_quil_via_entry_point(self):
        """Test that a quil string can be loaded via the load entrypoint."""
        quil_str = textwrap.dedent(
            """
            H 0
            RZ(0.34) 1
            CNOT 0 3
            H 2
            H 7
            X 7
            Y 1
            RZ(0.34) 1
        """
        )

        with OperationRecorder() as rec:
            qml.load(quil_str, format="quil")(wires=range(5))

        # The wires should be assigned as
        # 0  1  2  3  7
        # 0  1  2  3  4

        expected_queue = [
            qml.Hadamard(0),
            qml.RZ(0.34, wires=[1]),
            qml.CNOT(wires=[0, 3]),
            qml.Hadamard(2),
            qml.Hadamard(4),
            qml.PauliX(4),
            qml.PauliY(1),
            qml.RZ(0.34, wires=[1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    def test_load_quil_file_via_entry_point(self):
        """Test that a quil file can be loaded via the load entrypoint."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        with OperationRecorder() as rec:
            qml.load(os.path.join(cur_dir, "simple_program.quil"), format="quil_file")(
                wires=range(5)
            )

        # The wires should be assigned as
        # 0  1  2  3  7
        # 0  1  2  3  4

        expected_queue = [
            qml.Hadamard(0),
            qml.RZ(0.34, wires=[1]),
            qml.CNOT(wires=[0, 3]),
            qml.Hadamard(2),
            qml.Hadamard(4),
            qml.PauliX(4),
            qml.PauliY(1),
            qml.RZ(0.34, wires=[1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.data == expected.data

    @pytest.mark.parametrize("angle", [0.0, 0.3, 0.5, 0.7, -0.2, 2.4])
    def test_program_in_qnode(self, angle):
        """Test how a simple program is used inside a QNode."""
        program = pyquil.Program()

        delta = program.declare("delta", "REAL")

        program += g.RX(delta, 0)
        program += g.CNOT(0, 1)

        loader = qml.load(program, format="pyquil_program")
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(a):
            loader(wires=[0, 1], parameter_map={delta: a})

            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        @qml.qnode(dev)
        def circuit_reg(a):
            qml.RX(a, wires=[0])
            qml.CNOT(wires=[0, 1])

            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.array_equal(circuit(angle), circuit_reg(angle))

    @pytest.mark.parametrize("angle", [0.0, 0.3, 0.5, 0.7, -0.2, 2.4])
    def test_quil_in_qnode(self, angle):
        """Test how a simple quil string is used inside a QNode."""
        quil_str = textwrap.dedent(
            """
            DECLARE delta REAL[1]
            RX(delta) 0
            CNOT 0 1
        """
        )

        loader = qml.load(quil_str, format="quil")
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(a):
            loader(wires=[0, 1], parameter_map={"delta": a})

            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        @qml.qnode(dev)
        def circuit_reg(a):
            qml.RX(a, wires=[0])
            qml.CNOT(wires=[0, 1])

            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.array_equal(circuit(angle), circuit_reg(angle))

    @pytest.mark.parametrize("angle", [0.0, 0.3, 0.5, 0.7, -0.2, 2.4])
    def test_differentiation_in_qnode(self, angle):
        """Test a quil string used in a QNode can be differentiated."""
        quil_str = textwrap.dedent(
            """
            DECLARE delta REAL[1]
            RX(delta) 0
            CNOT 0 1
        """
        )

        loader = qml.load(quil_str, format="quil")
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(a):
            loader(wires=[0, 1], parameter_map={"delta": a})

            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        @qml.qnode(dev)
        def circuit_reg(a):
            qml.RX(a, wires=[0])
            qml.CNOT(wires=[0, 1])

            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.array_equal(qml.jacobian(circuit)(angle), qml.jacobian(circuit_reg)(angle))

    THETA = np.linspace(0.11, 3, 5)
    PHI = np.linspace(0.32, 3, 5)
    VARPHI = np.linspace(0.02, 3, 5)

    @pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
    def test_gradient(self, theta, phi, varphi, tol):
        """Test that the gradient works correctly"""
        program = pyquil.Program()

        delta1 = program.declare("delta1", "REAL")
        delta2 = program.declare("delta2", "REAL")
        delta3 = program.declare("delta3", "REAL")

        program += g.RX(delta1, 0)
        program += g.RX(delta2, 1)
        program += g.RX(delta3, 2)
        program += g.CNOT(0, 1)
        program += g.CNOT(1, 2)

        # Convert to a PennyLane circuit
        program_pl = qml.load(program, format="pyquil_program")

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(params):
            program_pl(parameter_map={delta1: params[0],
                                      delta2: params[1],
                                      delta3: params[2]},
                                      wires=range(len(program_pl.defined_qubits)))
            return qml.expval(qml.PauliX(0) @ qml.PauliY(2))

        dcircuit = qml.grad(circuit, 0)
        res = dcircuit([theta, phi, varphi])
        expected = [
            np.cos(theta) * np.sin(phi) * np.sin(varphi),
            np.sin(theta) * np.cos(phi) * np.sin(varphi),
            np.sin(theta) * np.sin(phi) * np.cos(varphi)
        ]

        assert np.allclose(res, expected, tol)
