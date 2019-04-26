.. _usage:

Plugin usage
############

PennyLane-Forest provides four Forest devices for PennyLane:

* :class:`forest.numpy_wavefunction <~NumpyWavefunctionDevice>`: provides a PennyLane device for the pyQVM Numpy wavefunction simulator

* :class:`forest.wavefunction <~WavefunctionDevice>`: provides a PennyLane device for the Forest wavefunction simulator

* :class:`forest.qvm <~QVMDevice>`: provides a PennyLane device for the Forest QVM and pyQuil pyQVM simulator

* :class:`forest.qpu <~QPUDevice>`: provides a PennyLane device for Forest QPU hardware devices


Using the devices
=================

Once PyQuil and the PennyLane plugin are installed, the three Forest devices can be accessed straight away in PennyLane.

You can instantiate these devices in PennyLane as follows:

>>> import pennylane as qml
>>> dev_numpy = qml.device('forest.numpy_wavefunction', wires=2)
>>> dev_simulator = qml.device('forest.wavefunction', wires=2)
>>> dev_pyqvm = qml.device('forest.qvm', device='2q-pyqvm', shots=1000)
>>> dev_qvm = qml.device('forest.qvm', device='2q-qvm', shots=1000)
>>> dev_qpu = qml.device('forest.qpu', device='Aspen-0-12Q-A', shots=1000)



These devices can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.

A simple quantum function that returns the expectation value of a measurement and depends on three classical input parameters would look like:

.. code-block:: python

    @qml.qnode(dev_qvm)
    def circuit(x, y, z):
        qml.RZ(z, wires=[0])
        qml.RY(y, wires=[0])
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval.PauliX(wires=1)

You can then execute the circuit like any other function to get the quantum mechanical expectation value:

>>> circuit(0.2, 0.1, 0.3)
-0.017578125

It is also easy to perform abstract calculations on a physical Forest QPU:

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
        return qml.expval.PauliZ(1)

Note that:

1. We import NumPy from PennyLane. This is a requirement, so that PennyLane can perform backpropagation in hybrid quantum-classical models. Alternatively, you may use the experimental PennyLane `PyTorch <https://pennylane.readthedocs.io/en/latest/code/interfaces/torch.html>`_ and `TensorFlow <https://pennylane.readthedocs.io/en/latest/code/interfaces/tfe.html>`_ interfaces.

2. Additional Quil gates not provided directly in PennyLane are importable from :mod:`~.ops`. In this case, we import the :class:`~.PSWAP` gate.

We can then make use of the quantum hardware and PennyLane's automatic differentiation to determine analytic gradients:

>>> func(0.4, 0.1)
0.92578125
>>> df = qml.grad(func, argnum=0)
>>> df(0.4, 0.1)
-0.4130859375

For more complicated examples using the provided PennyLane optimizers for machine learning, check out the `PennyLane tutorials and Jupyter notebooks <https://pennylane.readthedocs.io/en/latest/tutorials/notebooks.html>`_.

See below for more details on using the provided Forest devices.


Device options
==============

On initialization, the PennyLane-Forest devices accept additional keyword arguments beyond the PennyLane default device arguments.

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
    ``qvm -S`` and ``quilc -S``), then you will not need to set these keyword arguments.

    Likewise, if you are running PennyLane using the Rigetti Quantum Cloud Service (QCS)
    on a provided QMI, these environment variables are set automatically and will also
    not need to be passed in PennyLane.


The ``forest.numpy_wavefunction`` device
========================================

The ``forest.numpy_wavefunction`` device provides an interface between PennyLane and the pyQuil `NumPy wavefunction simulator <http://docs.rigetti.com/en/stable/wavefunction_simulator.html>`_. Because the NumPy wavefunction simulator allows access and manipulation of the underlying quantum state vector, ``forest.numpy_wavefunction`` is able to support the full suite of PennyLane and Quil quantum operations and expectation values.

In addition, it is generally faster than running equivalent simulations on the QVM, as the final state can be inspected and the expectation value calculated analytically, rather than by sampling measurements.


.. note::

    Since the NumPy wavefunction simulator is written entirely in NumPy, no external
    Quil compiler is required.


.. note::

    By default, ``forest.numpy_wavefunction`` is initialized with ``shots=0``, indicating
    that the exact analytic expectation value is to be returned.

    If the number of trials or shots provided to the ``forest.numpy_wavefunction`` is
    instead non-zero, a spectral decomposition is performed and a Bernoulli distribution
    is constructed and sampled. This allows the ``forest.numpy_wavefunction`` device to
    'approximate' the effect of sampling the expectation value.


The ``forest.wavefunction`` device
==================================

The ``forest.wavefunction`` device provides an interface between PennyLane and the Forest SDK `wavefunction simulator <http://docs.rigetti.com/en/stable/wavefunction_simulator.html>`_. Because the wavefunction simulator allows access and manipulation of the underlying quantum state vector, ``forest.wavefunction`` is able to support the full suite of PennyLane and Quil quantum operations and expectation values.

In addition, it is generally faster than running equivalent simulations on the QVM, as the final state can be inspected and the expectation value calculated analytically, rather than by sampling measurements.

.. note::

    By default, ``forest.wavefunction`` is initialized with ``shots=0``, indicating
    that the exact analytic expectation value is to be returned.

    If the number of trials or shots provided to the ``forest.wavefunction`` is
    instead non-zero, a spectral decomposition is performed and a Bernoulli distribution
    is constructed and sampled. This allows the ``forest.wavefunction`` device to
    'approximate' the effect of sampling the expectation value.


The ``forest.qvm`` device
=========================

The ``forest.qvm`` device provides an interface between PennyLane and the Forest SDK `quantum virtual machine <http://docs.rigetti.com/en/stable/qvm.html>`_ or the pyQuil built-in pyQVM. The QVM is used to simulate various quantum abstract machines, ranging from simulations of physical QPUs to completely connected lattices.

Note that, unlike ``forest.wavefunction``, you do not pass the number of wires - this is inferred automatically from the requested quantum computer topology.

>>> dev = qml.device('forest.qvm', device='Aspen-1-16Q-A')
>>> dev.num_wires
16

In addition, you may also request a QVM with noise models to better simulate a physical QPU; this is done by passing the keyword argument ``noisy=True``:

>>> dev = qml.device('forest.qvm', device='Aspen-1-16Q-A', noisy=True)

Note that only the `default noise models <http://docs.rigetti.com/en/stable/noise.html>`_ provided by pyQuil are currently supported.

To specify the pyQVM, simply append ``pyqvm`` to the end of the device name instead of ``qvm``:

>>> dev = qml.device('forest.qvm', device='4q-pyqvm')


Choosing the quantum computer
-----------------------------

When initializing the ``forest.qvm`` device, the following required keyword argument must also be passed:

``device`` (*str* or *networkx.Graph*)
    The name or topology of the quantum computer to initialize.

    * ``Nq-qvm``: for a fully connected/unrestricted N-qubit QVM
    * ``9q-qvm-square``: a :math:`9\times 9` lattice.
    * ``Nq-pyqvm`` or ``9q-pyqvm-square``, for the same as the above but run
       via the built-in pyQuil pyQVM device.
    * Any other supported Rigetti device architecture, for
      example a QPU lattice such as ``'Aspen-1-16Q-A'``.
    * Graph topology (as a ``networkx.Graph`` object) representing the device architecture.


Measurements and expectations
-----------------------------

Since the QVM returns a number of trial measurements of the quantum circuit, the larger the number of 'trials' or 'shots', the closer PennyLane is able to approximate the expectation value, and as a result the gradient. By default, ``shots=1024``, but this can be increased or decreased as required.

For example, see how increasing the shot count increases the expectation value and corresponding gradient accuracy:

.. code-block:: python

    def circuit(x):
        qml.RX(x, wires=[0])
        return qml.expval.PauliZ(0)

    dev_exact = qml.device('forest.wavefunction', wires=1)
    dev_s1024 = qml.device('forest.qvm', device='1q-qvm')
    dev_s100000 = qml.device('forest.qvm', device='1q-qvm', shots=100000)

    circuit_exact = qml.QNode(circuit, dev_exact)
    circuit_s1024 = qml.QNode(circuit, dev_s1024)
    circuit_s100000 = qml.QNode(circuit, dev_s100000)

Printing out the results of the three device expectation values:

>>> circuit_exact(0.8)
0.6967067093471655
>>> circuit_s1024(0.8)
0.689453125
>>> circuit_s100000(0.8)
0.6977


Supported expectation values
----------------------------

The QVM device supports ``qml.expval.PauliZ`` expectation values 'natively', while also supporting ``qml.expval.Identity``, ``qml.expval.PauliY``, ``qml.expval.Hadamard``, and ``qml.expval.Hermitian`` by performing implicit change of basis operations.

Native expectation values
^^^^^^^^^^^^^^^^^^^^^^^^^

The QVM currently supports only one measurement, returning ``1`` if the qubit is measured to be in the state :math:`|1\rangle`, and ``0`` if the qubit is measured to be in the state :math:`|0\rangle`. This is equivalent to measuring in the Pauli-Z basis, with state :math:`|1\rangle` corresponding to Pauli-Z eigenvalue :math:`\lambda=-1`, and likewise state :math:`|0\rangle` corresponding to eigenvalue :math:`\lambda=1`. As a result, we can simply perform a rescaling of the measurement results to get the Pauli-Z expectation value of the :math:`i` th wire:

.. math::
    \langle Z \rangle_{i} = \frac{1}{N}\sum_{j=1}^N (1-2m_j)

where :math:`N` is the total number of shots, and :math:`m_j` is the :math:`j` th measurement of wire :math:`i`.

Change of measurement basis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the remaining expectation values, it is easy to perform a quantum change of basis operation before measurement such that the correct expectation value is performed. For example, say we have a unitary Hermitian observable :math:`\hat{A}`. Since, by definition, it must have eigenvalues :math:`\pm 1`, there will always exist a unitary matrix :math:`U` such that it satisfies the following similarity transform:

.. math:: \hat{A} = U^\dagger Z U

Since :math:`U` is unitary, it can be applied to the specified qubit before measurement in the Pauli-Z basis. Below is a table of the various change of basis operations performed implicitly by PennyLane.

+-------------------------+-----------------------------------+
|    Expectation value    | Change of basis gate    :math:`U` |
+=========================+===================================+
| ``qml.expval.PauliX``   | :math:`H`                         |
+-------------------------+-----------------------------------+
| ``qml.expval.PauliY``   | :math:`H S^{-1}=HSZ`              |
+-------------------------+-----------------------------------+
| ``qml.expval.Hadamard`` | :math:`R_y(-\pi/4)`               |
+-------------------------+-----------------------------------+

To see how this affects the resultant quil program, you may use the :attr:`~.ForestDevice.program` property to print out the quil program after evaluation on the device.

.. code-block:: python

    dev = qml.device('forest.qvm', device='2q-qvm')

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=[0])
        return qml.expval.PauliY(0)

>>> circuit(0.54)
-0.525390625
>>> print(dev.program)
PRAGMA INITIAL_REWIRING "PARTIAL"
RX(0.54000000000000004) 0
Z 0
S 0
H 0
DECLARE ro BIT[1]
MEASURE 0 ro[0]

.. note::

    :attr:`~.ForestDevice.program` will return the **last evaluated quantum program** performed on the device. If viewing :attr:`~.ForestDevice.program` after evaluating a quantum gradient or performing an optimization, this may not match the user-defined QNode, as PennyLane automatically modifies the QNode to take into account the `parameter shift rule <https://pennylane.readthedocs.io/en/latest/concepts/autograd_quantum.html>`_, product rule, and chain rule.


Arbitrary Hermitian observables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Arbitrary Hermitian expectation values, ``qml.expval.Hermitian``, are also supported by the QVM. However, since they are not necessarily unitary (and thus have eigenvalues :math:`\lambda_i\neq \pm 1`), we cannot use the similarity transform approach above.

Instead, we can calculate the eigenvectors :math:`\mathbf{v}_i` of :math:`\hat{A}`, and construct our unitary change of basis operation as follows:

.. math:: U=\begin{bmatrix}\mathbf{v}_1 & \mathbf{v}_2 \end{bmatrix}^\dagger.

After measuring the qubit state, we can determine the probability :math:`P_0` of measuring state :math:`|0\rangle` and the probability :math:`P_1` of measuring state :math:`|1\rangle`, and, using the eigenvalues of :math:`\hat{A}`, recover the expectation value :math:`\langle\hat{A}\rangle`:

.. math:: \langle\hat{A}\rangle = \lambda_1 P_0 + \lambda_2 P_1


This process is done automatically behind the scenes in the QVM device when ``qml.expval.Hermitian`` is returned.




The ``forest.qpu`` device
=========================

The intention of the ``forest.qpu`` device is to construct a device that will allow for execution on an actual QPU. Constructing and using this device is very similar to very similar in design and implementation as the ``forest.qvm`` device, with slight differences at initialization, such as not supporting the keyword argument ``noisy``.

In addition, ``forest.qpu`` also accepts the optional ``active_reset`` keyword argument:

``active_reset`` (*bool*)
    Whether to actively reset qubits instead of waiting for
    for qubits to decay to the ground state naturally. Default is ``False``.
    Setting this to ``True`` results in a significantly faster expectation value
    evaluation when the number of shots is larger than ~1000.


Supported operations
====================

All devices support all PennyLane `operations <https://pennylane.readthedocs.io/en/latest/code/ops/qubit.html>`_ and `expectation <https://pennylane.readthedocs.io/en/latest/code/expval/qubit.html>`_ values, with the exception of the PennyLane ``QubitStateVector`` state preparation operation.

In addition, PennyLane-Forest provides the following PyQuil-specific operations for PennyLane. These are all importable from :mod:`pennylane_forest.ops <.ops>`.

These operations include:

.. autosummary::
    pennylane_forest.ops.S
    pennylane_forest.ops.T
    pennylane_forest.ops.CCNOT
    pennylane_forest.ops.CPHASE
    pennylane_forest.ops.CSWAP
    pennylane_forest.ops.ISWAP
    pennylane_forest.ops.PSWAP
