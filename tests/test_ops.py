import numpy as np
import pennylane as qml
import pennylane.devices.default_qubit as dq
import pytest
from pennylane_forest.ops import *
from pennylane.wires import Wires


class TestDecompositions:
    """Test that the gate decompositions are correct."""

    @pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, num=10))
    @pytest.mark.parametrize("q", [0, 1, 2, 3])
    def test_cphase_decomposition(self, phi, q, tol):
        """Test that the decomposition of CPHASE is correct."""
        expected_diagonal = np.array([1, 1, 1, 1], dtype=complex)
        expected_diagonal[q] = np.exp(1j * phi)
        expected_matrix = np.diag(expected_diagonal)

        decomposition = CPHASE.decomposition(phi, q, wires=[0, 1])

        calculated_matrix = np.eye(4, dtype=complex)
        default_qubit = qml.device("default.qubit", wires=2)
        for gate in decomposition:
            gate_matrix = qml.matrix(gate)

            if gate.num_wires == 1:
                if gate.wires[0] == 0:
                    gate_matrix = np.kron(gate_matrix, np.eye(2, dtype=complex))
                elif gate.wires[0] == 1:
                    gate_matrix = np.kron(np.eye(2, dtype=complex), gate_matrix)

            calculated_matrix = gate_matrix @ calculated_matrix

        assert np.allclose(expected_matrix, calculated_matrix, atol=tol, rtol=0)
