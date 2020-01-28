"""
Unit tests for the abstract ForestDevice class
"""
from unittest.mock import patch
import pytest

import pennylane as qml
from pennylane import numpy as np

from conftest import I, U, U2, H

from pennylane_forest.device import ForestDevice


# make test deterministic
np.random.seed(42)


@patch.multiple(ForestDevice, __abstractmethods__=set())
class TestMatVecProduct:
    """Unit tests matrix-vector product method of the ForestDevice class"""

    def test_incorrect_matrix_size(self, tol):
        """Test that an exception is raised if the input matrix is
        applied to the incorrect number of wires"""
        wires = 3
        dev = ForestDevice(wires=wires, shots=1)

        # create a random length 2**wires vector
        vec = np.random.random([2 ** wires])

        # apply 2 qubit unitary to the full system
        with pytest.raises(ValueError, match="specify a 8 x 8 matrix for 3 wires"):
            res = dev.mat_vec_product(U2, vec, wires=[0, 1, 2])

    def test_full_system(self, tol):
        """Test that matrix-vector multiplication
        over the entire subsystem agrees with standard
        dense matrix multiplication"""
        wires = 3
        dev = ForestDevice(wires=wires, shots=1)

        # create a random length 2**wires vector
        vec = np.random.random([2 ** wires])

        # make a 3 qubit unitary
        mat = np.kron(U2, H)

        # apply to the system
        res = dev.mat_vec_product(mat, vec, wires=[0, 1, 2])

        # perform the same operation using dense matrix multiplication
        expected = mat @ vec

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_permutation_full_system(self, tol):
        """Test that matrix-vector multiplication
        over a permutation of the system agrees with standard
        dense matrix multiplication"""
        wires = 2
        dev = ForestDevice(wires=wires, shots=1)

        # create a random length 2**wires vector
        vec = np.random.random([2 ** wires])

        # apply to the system
        res = dev.mat_vec_product(U2, vec, wires=[1, 0])

        # perform the same operation using dense matrix multiplication
        perm = np.array([0, 2, 1, 3])
        expected = U2[:, perm][perm] @ vec

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_consecutive_subsystem(self, tol):
        """Test that matrix-vector multiplication
        over a consecutive subset of the system agrees with standard
        dense matrix multiplication"""
        wires = 3
        dev = ForestDevice(wires=wires, shots=1)

        # create a random length 2**wires vector
        vec = np.random.random([2 ** wires])

        # apply a 2 qubit unitary to wires 1, 2
        res = dev.mat_vec_product(U2, vec, wires=[1, 2])

        # perform the same operation using dense matrix multiplication
        expected = np.kron(I, U2) @ vec

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_non_consecutive_subsystem(self, tol):
        """Test that matrix-vector multiplication
        over a non-consecutive subset of the system agrees with standard
        dense matrix multiplication"""
        wires = 3
        dev = ForestDevice(wires=wires, shots=1)

        # create a random length 2**wires vector
        vec = np.random.random([2 ** wires])

        # apply a 2 qubit unitary to wires 1, 2
        res = dev.mat_vec_product(U2, vec, wires=[2, 0])

        # perform the same operation using dense matrix multiplication
        perm = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        expected = np.kron(U2, I)[:, perm][perm] @ vec

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_apply_basis_state_raises_an_error_if_not_first(self):
        """Test that there is an error raised when the BasisState is not
        applied as the first operation."""
        wires = 3
        dev = ForestDevice(wires=wires, shots=1)

        operation = qml.BasisState(np.array([1,0,0]), wires=list(range(wires)))
        queue = [qml.PauliX(0), operation]
        with pytest.raises(qml.DeviceError, match="Operation {} cannot be used after other Operations have already been applied".format(operation.name)):
            dev.apply(queue)

    def test_apply_qubitstatesvector_raises_an_error_if_not_first(self):
        """Test that there is an error raised when the QubitStateVector is not
        applied as the first operation."""
        wires = 1
        dev = ForestDevice(wires=wires, shots=1)

        operation = qml.QubitStateVector(np.array([1,0]), wires=list(range(wires)))
        queue = [qml.PauliX(0), operation]
        with pytest.raises(qml.DeviceError, match="Operation {} cannot be used after other Operations have already been applied".format(operation.name)):
            dev.apply(queue)
