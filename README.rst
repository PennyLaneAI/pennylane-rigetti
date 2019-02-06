PennyLane Forest Plugin
#######################


This PennyLane plugin allows the Rigetti Forest and pyQuil simulators to be used as PennyLane devices.

`pyQuil <https://pyquil.readthedocs.io>`_ is a Python library for quantum programming using the quantum instruction language (Quil) - resulting quantum programs can be executed using the `Rigetti Forest SDK <https://www.rigetti.com/forest>`_ and the `Rigetti QCS <https://www.rigetti.com/qcs>`_.

`PennyLane <https://pennylane.readthedocs.io>`_ is a machine learning library for optimization and automatic differentiation of hybrid quantum-classical computations.


Features
========

* Provides three devices to be used with PennyLane: ``forest.wavefunction``, ``forest.qvm``, and ``forest.qpu``. These provide access to the Forest wavefunction simulator, quantum virtual machine (QVM), and quantum processing unit (QPU) respectively.

* All provided devices support all core qubit PennyLane operations.

* In addition, ``forest.wavefunction`` supports all core qubit PennyLane expectation values. ``forest.qvm`` and ``forest.qpu`` only support ``PauliZ`` expectation values.

* Provides custom PennyLane operations to cover additional pyQuil operations: ``T``, ``S``, ``ISWAP``, ``CCNOT``, ``PSWAP``, and many more. Every custom operation supports analytic differentiation.

* Combine Forest and the Rigetti Cloud Services with PennyLane's automatic differentiation and optimization.


Installation
============

PennyLane-Forest requires both PennyLane and pyQuil. It can be installed via ``pip``:

.. code-block:: bash

    $ python -m pip install pennylane-forest


Getting started
===============

Once the PennyLane-Forest plugin is installed, the two provided pyQuil devices can be accessed straight away in PennyLane.

You can instantiate these devices for PennyLane as follows:

.. code-block:: python

    import pennylane as qml
    dev_simulator = qml.device('forest.wavefunction', wires=2)
    dev_qvm = qml.device('forest.qvm', device='complete', wires=2, shots=1000)
    dev_qpu = qml.device('forest.qpu', device='Aspen-0-12Q-A', shots=1000)

These devices can then be used just like other devices for the definition and evaluation of QNodes within PennyLane. For more details, see the `plugin usage guide <https://pennylane-forest.readthedocs.io/en/latest/usage.html>`_ and refer to the PennyLane documentation.


Contributing
============

We welcome contributions - simply fork the PennyLane-Forest repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.  All contributers to PennyLane-Forest will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects or applications built on PennyLane and pyQuil.


Authors
=======

`Josh Izaac <https://github.com/josh146>`_, `Keri A. McKiernan <https://github.com/kmckiern>`_


Support
=======

- **Source Code:** https://github.com/rigetti/pennylane-forest
- **Issue Tracker:** https://github.com/rigetti/pennylane-forest/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.


License
=======

PennyLane-Forest is **free** and **open source**, released under the BSD 3-Clause license.
