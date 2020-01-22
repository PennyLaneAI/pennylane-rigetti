"""
Default parameters, commandline arguments and common routines for the unit tests.
"""
import pytest

from requests import RequestException

import numpy as np
from scipy.linalg import expm, block_diag

from pyquil import get_qc, Program
from pyquil.gates import I as Id
from pyquil.api import QVMConnection, QVMCompiler, local_qvm
from pyquil.api._config import PyquilConfig
from pyquil.api._errors import UnknownApiError
from pyquil.api._qvm import QVMNotRunning


# defaults
TOLERANCE = 1e-5
QVM_SHOTS = 10000


# pyquil specific global variables and functions
I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)


# U2 = np.array([[-0.07843244-3.57825948e-01j, 0.71447295-5.38069384e-02j, 0.20949966+6.59100734e-05j, -0.50297381+2.35731613e-01j],
#                [-0.26626692+4.53837083e-01j, 0.27771991-2.40717436e-01j, 0.41228017-1.30198687e-01j, 0.01384490-6.33200028e-01j],
#                [-0.69254712-2.56963068e-02j, -0.15484858+6.57298384e-02j, -0.53082141+7.18073414e-02j, -0.41060450-1.89462315e-01j],
#                [-0.09686189-3.15085273e-01j, -0.53241387-1.99491763e-01j, 0.56928622+3.97704398e-01j, -0.28671074-6.01574497e-02j]])


U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)


U_toffoli = np.diag([1 for i in range(8)])
U_toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])


H = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])


def controlled_phase(phi, q):
    """Returns the matrix representation of the controlled phase gate"""
    mat = np.identity(4, dtype=np.complex128)
    mat[q, q] = np.exp(1j * phi)
    return mat


test_operation_map = {
    "Identity": I,
    "PauliX": X,
    "PauliY": Y,
    "PauliZ": Z,
    "Hadamard": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "CNOT": block_diag(I, X),
    "SWAP": SWAP,
    "CZ": block_diag(I, Z),
    "PhaseShift": lambda phi: np.array([[1, 0], [0, np.exp(1j * phi)]]),
    "RX": lambda phi: expm(-1j * phi / 2 * X),
    "RY": lambda phi: expm(-1j * phi / 2 * Y),
    "RZ": lambda phi: expm(-1j * phi / 2 * Z),
    "Rot": lambda a, b, c: expm(-1j * c / 2 * Z) @ expm(-1j * b / 2 * Y) @ expm(-1j * a / 2 * Z),
    # arbitrary unitaries
    "Hermitian": H,
    "QubitUnitary": U,
    # the following gates are provided by the PL-Forest plugin
    "S": np.array([[1, 0], [0, 1j]]),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
    "Toffoli": block_diag(I, I, I, X),
    "CPHASE": controlled_phase,
    "CSWAP": block_diag(I, I, SWAP),
    "ISWAP": np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]),
    "PSWAP": lambda phi: np.array(
        [[1, 0, 0, 0], [0, 0, np.exp(1j * phi), 0], [0, np.exp(1j * phi), 0, 0], [0, 0, 0, 1]]
    ),
}


# command line arguments
def cmd_args(parser):
    """Command line argument parser"""
    parser.addoption("-t", "--tol", type=float, help="Numerical tolerance for equality tests.")


@pytest.fixture
def tol(request):
    """Tolerance fixture"""
    try:
        return request.config.getoption("--tol")
    except:
        return TOLERANCE


@pytest.fixture
def shots():
    """default shots"""
    return QVM_SHOTS


@pytest.fixture
def apply_unitary():
    """Apply a unitary operation to the ground state."""

    def _apply_unitary(mat, num_wires):
        """Applies a unitary to the first wire of a register in the ground state

        Args:
            mat (array): unitary matrix
            num_wires (n): number of wires in the register
        """
        N = mat.shape[0]

        init_state = np.zeros(N)
        init_state[0] = 1
        result = mat @ init_state

        for _ in range(num_wires - int(np.log2(N))):
            result = np.kron(result, np.array([1, 0]))

        return result

    return _apply_unitary


@pytest.fixture(scope="session")
def qvm():
    try:
        qvm = QVMConnection(random_seed=52)
        qvm.run(Program(Id(0)), [])
        return qvm
    except (RequestException, UnknownApiError, QVMNotRunning, TypeError) as e:
        return pytest.skip("This test requires QVM connection: {}".format(e))


@pytest.fixture(scope="session")
def compiler():
    try:
        config = PyquilConfig()
        device = get_qc("3q-qvm").device
        compiler = QVMCompiler(endpoint=config.quilc_url, device=device)
        compiler.quil_to_native_quil(Program(Id(0)))
        return compiler
    except (RequestException, UnknownApiError, QVMNotRunning, TypeError) as e:
        return pytest.skip("This test requires compiler connection: {}".format(e))


class BaseTest:
    """Default base test class"""

    # pylint: disable=no-self-use

    def assertEqual(self, first, second):
        """Replaces unittest TestCase.assertEqual"""
        assert first == second

    def assertAlmostEqual(self, first, second, delta):
        """Replaces unittest TestCase.assertEqual"""
        assert np.abs(first - second) <= delta

    def assertTrue(self, first):
        """Replaces unittest TestCase.assertTrue"""
        assert first

    def assertFalse(self, first):
        """Replaces unittest TestCase.assertFalse"""
        assert not first

    def assertAllAlmostEqual(self, first, second, delta):
        """
        Like assertAlmostEqual, but works with arrays. All the corresponding elements have to be almost equal.
        """
        if isinstance(first, tuple):
            # check each element of the tuple separately (needed for when the tuple elements are themselves batches)
            if np.all([np.all(first[idx] == second[idx]) for idx, _ in enumerate(first)]):
                return
            if np.all(
                [np.all(np.abs(first[idx] - second[idx])) <= delta for idx, _ in enumerate(first)]
            ):
                return
        else:
            if np.all(first == second):
                return
            if np.all(np.abs(first - second) <= delta):
                return
        assert False, "{} != {} within {} delta".format(first, second, delta)

    def assertAllEqual(self, first, second):
        """
        Like assertEqual, but works with arrays. All the corresponding elements have to be equal.
        """
        return self.assertAllAlmostEqual(first, second, delta=0.0)

    def assertAllTrue(self, value):
        """
        Like assertTrue, but works with arrays. All the corresponding elements have to be True.
        """
        return self.assertTrue(np.all(value))
