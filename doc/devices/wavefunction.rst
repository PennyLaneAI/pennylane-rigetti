The Wavefunction device
=======================

The ``forest.wavefunction`` device provides an interface between PennyLane and
the Forest SDK `wavefunction simulator <https://pyquil-docs.rigetti.com/en/stable/wavefunction_simulator.html>`_. Because
the wavefunction simulator allows access and manipulation of the underlying quantum state vector,
``forest.wavefunction`` is able to support the full suite of PennyLane and Quil quantum operations and observables.

In addition, it is generally faster than running equivalent simulations on the QVM, as the final state
can be inspected and the expectation value calculated analytically, rather than by sampling measurements.

.. note::

    By default, ``forest.wavefunction`` is initialized with ``shots=0``, indicating
    that the exact analytic expectation value is to be returned.

    If the number of trials or shots provided to the ``forest.wavefunction`` is
    instead non-zero, a spectral decomposition is performed and a Bernoulli distribution
    is constructed and sampled. This allows the ``forest.wavefunction`` device to
    'approximate' the effect of sampling the expectation value.

Usage
~~~~~

You can instantiate the device in PennyLane as follows:

.. code-block:: python

    import pennylane as qml

    dev_wfun = qml.device('forest.wavefunction', wires=2)
    
This device can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.

A simple quantum function that returns the expectation value and variance of a measurement and 
depends on three classical input parameters would look like:

.. code-block:: python

    @qml.qnode(dev_wfun)
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

All devices support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/stable/introduction/operations.html#qubit-operations>`_, with the exception of the PennyLane ``QubitStateVector`` state preparation operation.

.. include:: ./qvm_and_quilc_server_configuration.rst
