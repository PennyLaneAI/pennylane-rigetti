The QVM device
===============

The ``forest.qvm`` device provides an interface between PennyLane and the Forest
SDK `quantum virtual machine <https://pyquil-docs.rigetti.com/en/stable/qvm.html>`_ or the pyQuil built-in
pyQVM. The QVM is used to simulate various quantum abstract machines, ranging from simulations of
physical QPUs to completely connected lattices.

Usage
~~~~~

When initializing the ``forest.qvm`` device, the following required keyword argument must also be passed:

``device`` (*str* or *networkx.Graph*)
    The name or topology of the quantum computer to initialize.

    * ``Nq-qvm``: for a fully connected/unrestricted N-qubit QVM.
    * ``9q-square-qvm``: a :math:`9\times 9` lattice.
    * ``Nq-pyqvm`` or ``9q-square-pyqvm``, for the same as the above but run via the built-in pyQuil pyQVM device.
    * Any other supported Rigetti device architecture, for example a QPU lattice such as ``'Aspen-8'``.
    * Graph topology (as a ``networkx.Graph`` object) representing the device architecture.

Note that, unlike ``forest.wavefunction``, you do not pass the number of wires - this is inferred
automatically from the requested quantum computer topology.

>>> import pennylane as qml
>>> dev = qml.device('forest.qvm', device='Aspen-8')
>>> dev.num_wires
16

In addition, you may also request a QVM with noise models to better simulate a physical
QPU; this is done by passing the keyword argument ``noisy=True``:

>>> dev = qml.device('forest.qvm', device='Aspen-8', noisy=True)

Note that only the `default noise models <https://pyquil-docs.rigetti.com/en/stable/apidocs/noise.html>`_ provided by
pyQuil are currently supported.

To specify the pyQVM, simply append ``pyqvm`` to the end of the device name instead of ``qvm``:

>>> dev = qml.device('forest.qvm', device='4q-pyqvm')

The device can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.

A simple quantum function that returns the expectation value and variance of a measurement and 
depends on three classical input parameters would look like:

.. code-block:: python

    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.RZ(z, wires=[0])
        qml.RY(y, wires=[0])
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), var(qml.PauliZ(1))

You can then execute the circuit like any other function to get the quantum mechanical expectation value and variance:

>>> circuit(0.2, 0.1, 0.3)
array([0.97517033, 0.04904283])


Measurements and expectations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the QVM returns a number of trial measurements of the quantum circuit, the larger the number of
'trials' or 'shots', the closer PennyLane is able to approximate the expectation value,
and as a result the gradient. By default, ``shots=1024``, but this can be increased or decreased as required.

For example, see how increasing the shot count increases the expectation value and corresponding gradient accuracy:

.. code-block:: python

    def circuit(x):
        qml.RX(x, wires=[0])
        return qml.expval(qml.PauliZ(0))

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

Supported operations
~~~~~~~~~~~~~~~~~~~~

All devices support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/stable/introduction/operations.html#qubit-operations>`_, with the exception of the PennyLane ``QubitStateVector`` state preparation operation.

Supported observables
~~~~~~~~~~~~~~~~~~~~~

The QVM device supports ``qml.PauliZ`` observables values 'natively', while also supporting ``qml.Identity``, ``qml.PauliY``, ``qml.Hadamard``, and ``qml.Hermitian`` by performing implicit change of basis operations.

Native observables
^^^^^^^^^^^^^^^^^^

The QVM currently supports only one measurement, returning ``1`` if the qubit is measured to be in the state :math:`|1\rangle`, and ``0`` if the qubit is measured to be in the state :math:`|0\rangle`. This is equivalent to measuring in the Pauli-Z basis, with state :math:`|1\rangle` corresponding to Pauli-Z eigenvalue :math:`\lambda=-1`, and likewise state :math:`|0\rangle` corresponding to eigenvalue :math:`\lambda=1`. As a result, we can simply perform a rescaling of the measurement results to get the Pauli-Z expectation value of the :math:`i` th wire:

.. math::
    \langle Z \rangle_{i} = \frac{1}{N}\sum_{j=1}^N (1-2m_j)

where :math:`N` is the total number of shots, and :math:`m_j` is the :math:`j` th measurement of wire :math:`i`.

Change of measurement basis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the remaining observables, it is easy to perform a quantum change of basis operation before measurement such that the correct expectation value is performed. For example, say we have a unitary Hermitian observable :math:`\hat{A}`. Since, by definition, it must have eigenvalues :math:`\pm 1`, there will always exist a unitary matrix :math:`U` such that it satisfies the following similarity transform:

.. math:: \hat{A} = U^\dagger Z U

Since :math:`U` is unitary, it can be applied to the specified qubit before measurement in the Pauli-Z basis. Below is a table of the various change of basis operations performed implicitly by PennyLane.

+------------------+-----------------------------------+
|    Observable    | Change of basis gate    :math:`U` |
+==================+===================================+
| ``qml.PauliX``   | :math:`H`                         |
+------------------+-----------------------------------+
| ``qml.PauliY``   | :math:`H S^{-1}=HSZ`              |
+------------------+-----------------------------------+
| ``qml.Hadamard`` | :math:`R_y(-\pi/4)`               |
+------------------+-----------------------------------+

To see how this affects the resultant Quil program, you may use the :attr:`~.ForestDevice.program` property
to print out the Quil program after evaluation on the device.

.. code-block:: python

    dev = qml.device('forest.qvm', device='2q-qvm')

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=[0])
        return expval(qml.PauliY(0))

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

    :attr:`~.ForestDevice.program` will return the **last evaluated quantum program** performed on the device.
    If viewing :attr:`~.ForestDevice.program` after evaluating a quantum gradient or performing an optimization,
    this may not match the user-defined QNode, as PennyLane automatically modifies the QNode to take into account
    the `parameter shift rule <https://pennylane.ai/qml/glossary/parameter_shift.html>`_, product rule, and chain rule.


Arbitrary Hermitian observables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Arbitrary Hermitian observables, ``qml.Hermitian``, are also supported by the QVM. However, since they are not necessarily unitary (and thus have eigenvalues :math:`\lambda_i\neq \pm 1`), we cannot use the similarity transform approach above.

Instead, we can calculate the eigenvectors :math:`\mathbf{v}_i` of :math:`\hat{A}`, and construct our unitary change of basis operation as follows:

.. math:: U=\begin{bmatrix}\mathbf{v}_1 & \mathbf{v}_2 \end{bmatrix}^\dagger.

After measuring the qubit state, we can determine the probability :math:`P_0` of measuring state :math:`|0\rangle` and the probability :math:`P_1` of measuring state :math:`|1\rangle`, and, using the eigenvalues of :math:`\hat{A}`, recover the expectation value :math:`\langle\hat{A}\rangle`:

.. math:: \langle\hat{A}\rangle = \lambda_1 P_0 + \lambda_2 P_1


This process is done automatically behind the scenes in the QVM device when ``qml.expval(qml.Hermitian)`` is returned.

.. include:: ./qvm_and_quilc_server_configuration.rst
