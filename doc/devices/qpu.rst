The QPU device
==============

The intention of the ``forest.qpu`` device is to construct a device that will allow for execution on an actual QPU.
Constructing and using this device is very similar in design and implementation as the ``forest.qvm`` device, with 
slight differences at initialization, such as not supporting the keyword argument ``noisy``.

In addition, ``forest.qpu`` also accepts the optional ``active_reset`` keyword argument:

``active_reset`` (*bool*)
    Whether to actively reset qubits instead of waiting for
    for qubits to decay to the ground state naturally. Default is ``False``.
    Setting this to ``True`` results in a significantly faster expectation value
    evaluation when the number of shots is larger than ~1000.

Usage
~~~~~

A QPU device can be created via:

>>> import pennylane as qml
>>> dev_qpu = qml.device('forest.qpu', device='Aspen-M-2', shots=1000)

Note that additional Quil gates not provided directly in PennyLane are importable from :mod:`~.ops`.
An example that demonstrates the use of the native :class:`~.PSWAP` plugin gate is this:

.. code-block:: python

    from pennylane import numpy as np
    from pennylane_forest.ops import PSWAP

    @qml.qnode(dev_qpu)
    def func(x, y):
        qml.BasisState(np.array([1, 1]), wires=0)
        qml.RY(x, wires=0)
        qml.RX(y, wires=1)
        PSWAP(0.432, wires=[0, 1])
        qml.CNOT(wires=[0, 1])
        return expval(qml.PauliZ(1))

We can then integrate the quantum hardware and PennyLane's automatic differentiation to determine analytic gradients:

>>> func(0.4, 0.1)
0.92578125
>>> df = qml.grad(func, argnum=0)
>>> df(0.4, 0.1)
-0.4130859375

Supported operations
~~~~~~~~~~~~~~~~~~~~

All devices support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/stable/introduction/operations.html#qubit-operations>`_, with the exception of the PennyLane ``QubitStateVector`` state preparation operation.

quilc server configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    If using the downloadable Forest SDK with the default server configurations
    for the Quil compiler (i.e., ``quilc -R``), then no special configuration is needed.
    If using a non-default port or host for the server, see the 
    `pyQuil configuration documentation <https://pyquil-docs.rigetti.com/en/stable/advanced_usage.html#pyquil-configuration>`_
    for details on how to override the default values.
