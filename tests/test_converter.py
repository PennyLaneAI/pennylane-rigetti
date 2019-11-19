import pytest
import pennylane as qml
from pennylane.utils import OperationRecorder
from pennylane_forest.converter import *
import pyquil
import pyquil.gates as g

class TestProgramConverter:
    """Test that PyQuil Program instances are properly converted."""

    # "PHASE" : qml.PhaseShift,
    # "RX" : qml.RX,
    # "RY" : qml.RY,
    # "RZ" : qml.RZ,
    # "CRX" : qml.CRX,
    # "CRY" : qml.CRY,
    # "CRZ" : qml.CRZ,

    # # the following gates are provided by the PL-Forest plugin
    # "S" : plf.ops.S,
    # "T" : plf.ops.T,
    # "CCNOT" : plf.ops.CCNOT,
    # "CPHASE" : plf.ops.CPHASE,
    # "CSWAP" : plf.ops.CSWAP,
    # "ISWAP" : plf.ops.ISWAP,
    # "PSWAP" : plf.ops.PSWAP,

    @pytest.mark.parametrize("pyquil_operation,expected_pl_operation", [
        (g.H(0), qml.Hadamard(0)),
        (g.H(0).dagger(), qml.Hadamard(0).inv()),
        (g.H(0).dagger().dagger(), qml.Hadamard(0).inv().inv()),
        (g.X(0), qml.PauliX(0)),
        (g.X(0).dagger(), qml.PauliX(0).inv()),
        (g.X(0).dagger().dagger(), qml.PauliX(0).inv().inv()),
        (g.X(0).controlled(1), qml.CNOT(wires=[1, 0])),
        (g.X(0).controlled(1).dagger(), qml.CNOT(wires=[1, 0]).inv()),
        (g.X(0).controlled(1).dagger().dagger(), qml.CNOT(wires=[1, 0]).inv().inv()),
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
        (g.CNOT(0, 1).controlled(2).dagger().dagger(), plf.ops.CCNOT(wires=[2, 0, 1]).inv().inv()),
        (g.SWAP(0, 1), qml.SWAP(wires=[0, 1])),
        (g.SWAP(0, 1).dagger(), qml.SWAP(wires=[0, 1]).inv()),
        (g.SWAP(0, 1).dagger().dagger(), qml.SWAP(wires=[0, 1]).inv().inv()),
        (g.SWAP(0, 1).controlled(2), qml.CSWAP(wires=[2, 0, 1])),
        (g.SWAP(0, 1).controlled(2).dagger(), qml.CSWAP(wires=[2, 0, 1]).inv()),
        (g.SWAP(0, 1).controlled(2).dagger().dagger(), qml.CSWAP(wires=[2, 0, 1]).inv().inv()),
        (g.CZ(0, 1), qml.CZ(wires=[0, 1])),
        (g.CZ(0, 1).dagger(), qml.CZ(wires=[0, 1]).inv()),
        (g.CZ(0, 1).dagger().dagger(), qml.CZ(wires=[0, 1]).inv().inv()),
    ])
    def test_convert_operation(self, pyquil_operation, expected_pl_operation):
        program = pyquil.Program()

        program += pyquil_operation

        with OperationRecorder() as rec:
            load_program(program)

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
            load_program(program)

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
            load_program(program)

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
            load_program(program)

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
            load_program(program)

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
