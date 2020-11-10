The QPU device
==============

The intention of the ``forest.qpu`` device is to construct a device that will allow for execution on an actual QPU.
Constructing and using this device is very similar to very similar in design and implementation as the
``forest.qvm`` device, with slight differences at initialization, such as not supporting the keyword argument ``noisy``.

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
>>> dev_qpu = qml.device('forest.qpu', device='Aspen-8', shots=1000)

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

All devices support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/latest/code/ops/qubit.html>`_, with
he exception of the PennyLane ``QubitStateVector`` state preparation operation.

In addition, PennyLane-Forest provides the following PyQuil-specific operations for PennyLane.
These are all importable from :mod:`pennylane_forest.ops <.ops>`.

These operations include:

.. autosummary::
    pennylane_forest.ops.S
    pennylane_forest.ops.T
    pennylane_forest.ops.CCNOT
    pennylane_forest.ops.CPHASE
    pennylane_forest.ops.CSWAP
    pennylane_forest.ops.ISWAP
    pennylane_forest.ops.PSWAP

Device options
~~~~~~~~~~~~~~

On initialization, the PennyLane-Forest devices accept additional keyword 
arguments beyond the PennyLane default device arguments.

``forest_url`` (*str*)
    the Forest URL server. Can also be set by
    the environment variable ``FOREST_SERVER_URL``, or in the ``~/.qcs_config``
    configuration file. Default value is ``"https://forest-server.qcs.rigetti.com"``.

``qvm_url`` (*str*)
    the QVM server URL. Can also be set by the environment
    variable ``QVM_URL``, or in the ``~/.forest_config`` configuration file.
    Default value is ``"http://127.0.0.1:5000"``.

``compiler_url`` (*str*)
    the compiler server URL. Can also be set by the environment
    variable ``COMPILER_URL``, or in the ``~/.forest_config`` configuration file.
    Default value is ``"http://127.0.0.1:6000"``.

.. note::

    If using the downloadable Forest SDK with the default server configurations
    for the QVM and the Quil compiler (i.e., you launch them with the commands
    ``qvm -S`` and ``quilc -R``), then you will not need to set these keyword arguments.

    Likewise, if you are running PennyLane using the Rigetti Quantum Cloud Service (QCS)
    on a provided QMI, these environment variables are set automatically and will also
    not need to be passed in PennyLane.