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
        program += g.H(7)
        program += g.X(7).dagger()
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
            qml.PauliY(1),
            qml.RZ(0.34, wires=[1]),
        ]

        for converted, expected in zip(rec.queue, expected_queue):
            assert converted.name == expected.name
            assert converted.wires == expected.wires
            assert converted.params == expected.params

    # program += g.RY(-0.34, 5).controlled(2)
    # program += g.RY(-0.34, 2).controlled(1)
    # program += g.RZ(0.7, 1).dagger()
