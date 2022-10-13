The Numpy-Wavefunction device
=============================

The ``forest.numpy_wavefunction`` device provides an interface between PennyLane
and the pyQuil ``NumpyWavefunctionSimulator``.

As the NumPy wavefunction simulator allows access and manipulation of the underlying
quantum state vector, ``forest.numpy_wavefunction`` is able to support the full
suite of PennyLane and Quil quantum operations and observables.


.. note::

    Since the NumPy wavefunction simulator is written entirely in NumPy, no external
    Quil compiler is required.

.. note::

    By default, ``forest.numpy_wavefunction`` is initialized with ``analytic=True``, indicating
    that the exact analytic expectation value is to be returned.

    If the number of trials or shots provided to the ``forest.numpy_wavefunction`` is
    instead non-zero, a spectral decomposition is performed and a Bernoulli distribution
    is constructed and sampled. This allows the ``forest.numpy_wavefunction`` device to
    'approximate' the effect of sampling the expectation value.

Usage
~~~~~

You can instantiate the device in PennyLane as follows:

.. code-block:: python

    import pennylane as qml

    dev_numpy = qml.device('forest.numpy_wavefunction', wires=2)

This device can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.

A simple quantum function that returns the expectation value and variance of a measurement and
depends on three classical input parameters would look like:

.. code-block:: python

    @qml.qnode(dev_numpy)
    def circuit(x, y, z):
        qml.RZ(z, wires=[0])
        qml.RY(y, wires=[0])
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), var(qml.PauliZ(1))

You can then execute the circuit like any other function to get the quantum mechanical expectation value and variance:

>>> circuit(0.2, 0.1, 0.3)
array([0.97517033, 0.04904283])

Supported operations
~~~~~~~~~~~~~~~~~~~~

All Forest devices support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/stable/introduction/operations.html#qubit-operations>`_, with the exception of the PennyLane ``QubitStateVector`` state preparation operation.
