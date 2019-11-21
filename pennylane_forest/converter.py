from collections.abc import Sequence
import pennylane as qml
import pennylane_forest as plf
import pyquil
import pyquil.gates as g

pyquil_inv_operation_map = {
    "X": qml.PauliX,
    "Y": qml.PauliY,
    "Z": qml.PauliZ,
    "H": qml.Hadamard,
    "CNOT": qml.CNOT,
    "SWAP": qml.SWAP,
    "CZ": qml.CZ,
    "PHASE": qml.PhaseShift,
    "RX": qml.RX,
    "RY": qml.RY,
    "RZ": qml.RZ,
    # Those are not native Quil gates, but we have them here
    # to support CONTROLLED RX and the like
    "CRX": qml.CRX,
    "CRY": qml.CRY,
    "CRZ": qml.CRZ,
    # the following gates are provided by the PL-Forest plugin
    "S": plf.ops.S,
    "T": plf.ops.T,
    "CCNOT": plf.ops.CCNOT,
    "CPHASE": lambda *params, wires: plf.ops.CPHASE(*params, 3, wires=wires),
    "CPHASE00": lambda *params, wires: plf.ops.CPHASE(*params, 0, wires=wires),
    "CPHASE01": lambda *params, wires: plf.ops.CPHASE(*params, 1, wires=wires),
    "CPHASE10": lambda *params, wires: plf.ops.CPHASE(*params, 2, wires=wires),
    "CSWAP": plf.ops.CSWAP,
    "ISWAP": plf.ops.ISWAP,
    "PSWAP": plf.ops.PSWAP,
}

_control_map = {
    "X": "CNOT",
    "Z": "CZ",
    "CNOT": "CCNOT",
    "SWAP": "CSWAP",
    "RX": "CRX",
    "RY": "CRY",
    "RZ": "CRZ",
    "PHASE": "CPHASE",
}


def _resolve_operation_name(gate):
    name = gate.name

    for modifier in gate.modifiers:
        if modifier == "CONTROLLED":
            name = _control_map[name]

    return name


def _get_qubit_index(qubit):
    if isinstance(qubit, int):
        return qubit

    if isinstance(qubit, pyquil.quilatom.Qubit):
        return qubit.index

    raise Exception("I can't get that Qubit index.")


def load_program(program):
    """Load template from PyQuil Program instance."""

    program_qubits = program.get_qubits()
    print("program.qet_qubits() = ", program_qubits)
    qubit_to_wire_map = dict(zip(program_qubits, range(len(program_qubits))))
    print("qubit_to_wire_map = ", qubit_to_wire_map)

    def _qubits_to_wires(qubits):
        if isinstance(qubits, Sequence):
            return [qubit_to_wire_map[_get_qubit_index(qubit)] for qubit in qubits]

        return qubit_to_wire_map[_get_qubit_index(qubits)]

    for i, gate in enumerate(program.instructions):
        print("gate[{}] = {}".format(i, gate))
        print("gate.modifiers = ", gate.modifiers)

        if "FORKED" in gate.modifiers:
            raise qml.DeviceError(
                "Forked gates can not be imported into PennyLane, as this functionality is not supported. "
                + "Gate Nr. {}, {} was forked.".format(i + 1, gate)
            )

        pl_gate = pyquil_inv_operation_map[_resolve_operation_name(gate)]

        wires = _qubits_to_wires(gate.qubits)
        pl_gate_instance = pl_gate(*gate.params, wires=wires)

        if gate.modifiers.count("DAGGER") % 2 == 1:
            pl_gate_instance.inv()
