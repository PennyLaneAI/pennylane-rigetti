"""
Custom operations
=================

**Module name:** :mod:`pennylane_rigetti.ops`

.. currentmodule:: pennylane_rigetti.ops

Contains some additional PennyLane qubit operations.

Operations
----------

.. autosummary::

    CPHASE
    ISWAP
    PSWAP


Code details
~~~~~~~~~~~~
"""

import pennylane as qml
from pennylane.operation import Operation


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
