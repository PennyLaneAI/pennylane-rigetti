"""
Custom operations
=================

**Module name:** :mod:`pennylane_forest.ops`

.. currentmodule:: pennylane_forest.ops

Contains some additional PennyLane qubit operations.

Operations
----------

.. autosummary::
    S
    T
    CCNOT
    CPHASE
    CSWAP
    ISWAP
    PSWAP


Code details
~~~~~~~~~~~~
"""
import pennylane as qml
from pennylane.operation import Operation


# We keep the following definitions for compatibility
# as they are now part of PennyLane core
S = qml.S
T = qml.T
CSWAP = qml.CSWAP
CCNOT = qml.Toffoli


class CPHASE(Operation):
    r"""CHPASE(phi, q, wires)
    Controlled-phase gate.

    .. math::

        CPHASE_{ij}(phi, q) = \begin{cases}
            0, & i\neq j\\
            1, & i=j, i\neq q\\
            e^{i\phi}, & i=j=q
        \end{cases}\in\mathbb{C}^{4\times 4}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 2
    * Gradient recipe: :math:`\frac{d}{d\phi}CPHASE(\phi) = \frac{1}{2}\left[CPHASE(\phi+\pi/2)+CPHASE(\phi-\pi/2)\right]`
      Note that the gradient recipe only applies to parameter :math:`\phi`.
      Parameter :math:`q\in\mathbb{N}_0` and thus ``CPHASE`` can not be differentiated
      with respect to :math:`q`.

    Args:
        phi (float): the controlled phase angle
        q (int): an integer between 0 and 3 that corresponds to a state
            :math:`\{00, 01, 10, 11\}` on which the conditional phase
            gets applied
        wires (int): the subsystem the gate acts on
    """
    num_params = 2
    num_wires = 2
    par_domain = "R"
    grad_method = "A"

    def decomposition(phi, q, wires):
        if q == 0:
            return [
                qml.PauliX(wires[0]),
                qml.PauliX(wires[1]),
                qml.PhaseShift(phi / 2, wires=[wires[0]]),
                qml.PhaseShift(phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
                qml.PhaseShift(-phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
                qml.PauliX(wires[1]),
                qml.PauliX(wires[0]),
            ]

        elif q == 1:
            return [
                qml.PauliX(wires[0]),
                qml.PhaseShift(phi / 2, wires=[wires[0]]),
                qml.PhaseShift(phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
                qml.PhaseShift(-phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
                qml.PauliX(wires[0]),
            ]

        elif q == 2:
            return [
                qml.PauliX(wires[1]),
                qml.PhaseShift(phi / 2, wires=[wires[0]]),
                qml.PhaseShift(phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
                qml.PhaseShift(-phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
                qml.PauliX(wires[1]),
            ]

        elif q == 3:
            return [
                qml.PhaseShift(phi / 2, wires=[wires[0]]),
                qml.PhaseShift(phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
                qml.PhaseShift(-phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
            ]


class ISWAP(Operation):
    r"""ISWAP(wires)
    iSWAP gate.

    .. math:: ISWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & i & 0\\
            0 & i & 0 & 0\\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 2
    par_domain = None

    def decomposition(wires):
        return [
            qml.SWAP(wires=wires),
            qml.S(wires=[wires[0]]),
            qml.S(wires=[wires[1]]),
            qml.CZ(wires=wires),
        ]


class PSWAP(Operation):
    r"""PSWAP(wires)
    Phase-SWAP gate.

    .. math:: PSWAP(\phi) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & e^{i\phi} & 0\\
            0 & e^{i\phi} & 0 & 0\\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 3
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}PSWAP(\phi) = \frac{1}{2}\left[PSWAP(\phi+\pi/2)+PSWAP(\phi-\pi/2)\right]`


    Args:
        wires (int): the subsystem the gate acts on
        phi (float): the phase angle
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"

    def decomposition(phi, wires):
        return [
            qml.SWAP(wires=wires),
            qml.CNOT(wires=wires),
            qml.PhaseShift(phi, wires=[wires[1]]),
            qml.CNOT(wires=wires),
        ]
