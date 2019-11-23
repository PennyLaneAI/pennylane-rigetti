import numpy as np
import pennylane as qml
import pyquil
import pyquil.gates as g
import pytest
from pennylane.utils import OperationRecorder
from pennylane_forest.converter import *


class TestProgramConverter:
    """Test that PyQuil Program instances are properly converted."""

    @pytest.mark.parametrize(
        "pyquil_operation,expected_pl_operation",
        [
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
            (g.X(0).controlled(1).controlled(2), plf.ops.CCNOT(wires=[2, 1, 0])),
            (g.X(0).controlled(1).controlled(2).dagger(), plf.ops.CCNOT(wires=[2, 1, 0]).inv()),
            (
                g.X(0).controlled(1).controlled(2).dagger().dagger(),
                plf.ops.CCNOT(wires=[2, 1, 0]).inv().inv(),
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
            (g.CNOT(0, 1).controlled(2), plf.ops.CCNOT(wires=[2, 0, 1])),
            (g.CNOT(0, 1).controlled(2).dagger(), plf.ops.CCNOT(wires=[2, 0, 1]).inv()),
            (
                g.CNOT(0, 1).controlled(2).dagger().dagger(),
                plf.ops.CCNOT(wires=[2, 0, 1]).inv().inv(),
            ),
            (g.SWAP(0, 1), qml.SWAP(wires=[0, 1])),
            (g.SWAP(0, 1).dagger(), qml.SWAP(wires=[0, 1]).inv()),
            (g.SWAP(0, 1).dagger().dagger(), qml.SWAP(wires=[0, 1]).inv().inv()),
            (g.SWAP(0, 1).controlled(2), qml.CSWAP(wires=[2, 0, 1])),
            (g.SWAP(0, 1).controlled(2).dagger(), qml.CSWAP(wires=[2, 0, 1]).inv()),
            (g.SWAP(0, 1).controlled(2).dagger().dagger(), qml.CSWAP(wires=[2, 0, 1]).inv().inv()),
            (g.ISWAP(0, 1), plf.ops.ISWAP(wires=[0, 1])),
            (g.ISWAP(0, 1).dagger(), plf.ops.ISWAP(wires=[0, 1]).inv()),
            (g.ISWAP(0, 1).dagger().dagger(), plf.ops.ISWAP(wires=[0, 1]).inv().inv()),
            (g.PSWAP(0.3, 0, 1), plf.ops.PSWAP(0.3, wires=[0, 1])),
            (g.PSWAP(0.3, 0, 1).dagger(), plf.ops.PSWAP(0.3, wires=[0, 1]).inv()),
            (g.PSWAP(0.3, 0, 1).dagger().dagger(), plf.ops.PSWAP(0.3, wires=[0, 1]).inv().inv()),
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
            (g.CCNOT(0, 1, 2), plf.ops.CCNOT(wires=[0, 1, 2])),
            (g.CCNOT(0, 1, 2).dagger(), plf.ops.CCNOT(wires=[0, 1, 2]).inv()),
            (g.CCNOT(0, 1, 2).dagger().dagger(), plf.ops.CCNOT(wires=[0, 1, 2]).inv().inv()),
        ],
    )
    def test_convert_operation(self, pyquil_operation, expected_pl_operation):
        program = pyquil.Program()

        program += pyquil_operation

        with OperationRecorder() as rec:
            load_program(program)()

        assert rec.queue[0].name == expected_pl_operation.name
        assert rec.queue[0].wires == expected_pl_operation.wires
        assert rec.queue[0].params == expected_pl_operation.params

    def test_convert_simple_program(self):
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
            load_program(program)()

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
            assert converted.params == expected.params

    def test_convert_simple_program_with_parameters(self):
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

        variable_map = {
            "alpha": a,
            "beta": b,
            "gamma": c,
        }

        with OperationRecorder() as rec:
            load_program(program)(variable_map=variable_map)

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
            assert converted.params == expected.params

    def test_convert_simple_program_with_parameters_mixed_keys(self):
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

        variable_map = {
            "alpha": a,
            beta: b,
            gamma: c,
            "delta": d,
        }

        with OperationRecorder() as rec:
            load_program(program)(variable_map=variable_map)

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
            assert converted.params == expected.params

    def test_convert_simple_program_wire_assignment(self):
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
            assert converted.params == expected.params

    @pytest.mark.parametrize("wires", [[0, 1, 2, 3], [4, 5]])
    def test_convert_wire_error(self, wires):
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
            load_program(program)()

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
            assert converted.params == expected.params

    def test_convert_program_with_controlled_operations(self):
        program = pyquil.Program()

        program += g.RZ(0.34, 1)
        program += g.RY(0.2, 3).controlled(2)
        program += g.RX(0.4, 2).controlled(0)
        program += g.CNOT(1, 4)
        program += g.CNOT(1, 6).controlled(3)
        program += g.X(3).controlled(4).controlled(1)

        with OperationRecorder() as rec:
            load_program(program)()

        expected_queue = [
            qml.RZ(0.34, wires=[1]),
            qml.CRY(0.2, wires=[2, 3]),
            qml.CRX(0.4, wires=[0, 2]),
            qml.CNOT(wires=[1, 4]),
            plf.ops.CCNOT(wires=[3, 1, 5]),
            plf.ops.CCNOT(wires=[1, 4, 3]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.params == expected.params

    def test_convert_program_with_controlled_operations_not_in_pl_core(self, tol):
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
            load_program(program)()

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
            assert np.allclose(converted.params, expected.params, atol=tol, rtol=0)

    def test_convert_program_with_controlled_dagger_operations(self):
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
            load_program(program)()

        expected_queue = [
            plf.ops.CCNOT(wires=[2, 0, 1]),
            plf.ops.CCNOT(wires=[2, 0, 1]).inv(),
            plf.ops.CCNOT(wires=[2, 0, 1]).inv(),
            plf.ops.CCNOT(wires=[2, 0, 1]),
            qml.CRX(0.3, wires=[4, 3]),
            qml.CRX(0.2, wires=[4, 3]).inv(),
            qml.CRX(0.3, wires=[4, 3]).inv(),
            qml.CRX(0.2, wires=[4, 3]),
            plf.ops.CCNOT(wires=[1, 4, 2]),
            plf.ops.CCNOT(wires=[1, 4, 0]).inv(),
            plf.ops.CCNOT(wires=[1, 4, 0]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.params == expected.params

    def test_convert_program_with_defgates(self):
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
            load_program(program)()

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
            assert converted.params == expected.params

    def test_convert_program_with_controlled_defgates(self, tol):
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
            load_program(program)()

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
            assert np.allclose(converted.params, expected.params, atol=tol, rtol=0)

    def test_convert_program_with_defpermutationgates(self):
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
            load_program(program)()

        expected_queue = [
            qml.CNOT(wires=[0, 1]),
            qml.QubitUnitary(expected_matrix, wires=[0, 1]),
            qml.CNOT(wires=[0, 1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert np.array_equal(converted.params, expected.params)

    def test_convert_program_with_controlled_defpermutationgates(self):
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
            load_program(program)()

        expected_queue = [
            qml.CNOT(wires=[0, 1]),
            qml.QubitUnitary(expected_controlled_matrix, wires=[2, 0, 1]),
            qml.QubitUnitary(expected_controlled_matrix, wires=[0, 1, 2]),
            qml.CNOT(wires=[0, 1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert np.array_equal(converted.params, expected.params)

    def test_forked_gate_error(self):
        program = pyquil.Program()

        program += g.CNOT(0, 1)
        program += g.RX(0.3, 1).forked(2, [0.5])
        program += g.CNOT(0, 1)

        with pytest.raises(
            qml.DeviceError,
            match="Forked gates can not be imported into PennyLane, as this functionality is not supported",
        ):
            load_program(program)()
