import pennylane as qml
from pennylane.utils import OperationRecorder
from pennylane_forest.converter import *
import pyquil
import pyquil.gates as g

class TestProgramConverter:
    """Test that PyQuil Program instances are properly converted."""

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
