PennyLane Forest Plugin
#######################

.. image:: https://img.shields.io/github/workflow/status/PennyLaneAI/pennylane-forest/Tests/master?logo=github&style=flat-square
    :alt: GitHub Workflow Status (branch)
    :target: https://github.com/PennyLaneAI/pennylane-forest/actions?query=workflow%3ATests

.. image:: https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane-forest/master.svg?logo=codecov&style=flat-square
    :alt: Codecov coverage
    :target: https://codecov.io/gh/PennyLaneAI/pennylane-forest

.. image:: https://img.shields.io/codefactor/grade/github/PennyLaneAI/pennylane-forest/master?logo=codefactor&style=flat-square
    :alt: CodeFactor Grade
    :target: https://www.codefactor.io/repository/github/pennylaneai/pennylane-forest

.. image:: https://img.shields.io/readthedocs/pennylane-forest.svg?logo=read-the-docs&style=flat-square
    :alt: Read the Docs
    :target: https://pennylaneforest.readthedocs.io

.. image:: https://img.shields.io/pypi/v/pennylane-forest.svg?style=flat-square
    :alt: PyPI
    :target: https://pypi.org/project/pennylane-forest

.. image:: https://img.shields.io/pypi/pyversions/pennylane-forest.svg?style=flat-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/pennylane-forest

.. header-start-inclusion-marker-do-not-remove

Contains the PennyLane Forest plugin. This plugin allows three Rigetti devices to work with
PennyLane --- the wavefunction simulator, the Quantum Virtual Machine (QVM), and Quantum Processing
Units (QPUs).

`pyQuil <https://pyquil.readthedocs.io>`_ is a Python library for quantum programming using the
quantum instruction language (Quil) --- resulting quantum programs can be executed using the
`Rigetti Forest SDK <https://www.PennyLaneAI.com/forest>`_ and the `Rigetti QCS
<https://www.PennyLaneAI.com/qcs>`_.

`PennyLane <https://pennylane.ai>`_ is a machine learning library for optimization and automatic
differentiation of hybrid quantum-classical computations.


.. header-end-inclusion-marker-do-not-remove

Features
========

* Provides four devices to be used with PennyLane: ``forest.numpy_wavefunction``,
  ``forest.wavefunction``, ``forest.qvm``, and ``forest.qpu``. These provide access to the pyQVM
  Numpy wavefunction simulator, Forest wavefunction simulator, quantum virtual machine (QVM), and
  quantum processing unit (QPU) respectively.


* All provided devices support all core qubit PennyLane operations and observables.


* Provides custom PennyLane operations to cover additional pyQuil operations: ``T``, ``S``,
  ``ISWAP``, ``CCNOT``, ``PSWAP``, and many more. Every custom operation supports analytic
  differentiation.

* Combine Forest and the Rigetti Cloud Services with PennyLane's automatic differentiation and
  optimization.


.. installation-start-inclusion-marker-do-not-remove

Installation
============

PennyLane-Forest requires both PennyLane and pyQuil. It can be installed via ``pip``:

.. code-block:: bash

    $ python -m pip install pennylane-forest


.. installation-end-inclusion-marker-do-not-remove

Getting started
===============

Once the PennyLane-Forest plugin is installed, the three provided pyQuil devices can be accessed straight away in PennyLane.

You can instantiate these devices for PennyLane as follows:

.. code-block:: python

    import pennylane as qml
    dev_numpy = qml.device('forest.numpy_wavefunction', wires=2)
    dev_simulator = qml.device('forest.wavefunction', wires=2)
    dev_pyqvm = qml.device('forest.qvm', device='2q-pyqvm', shots=1000)
    dev_qvm = qml.device('forest.qvm', device='2q-qvm', shots=1000)
    dev_qpu = qml.device('forest.qpu', device='Aspen-8', shots=1000)

These devices can then be used just like other devices for the definition and evaluation of QNodes
within PennyLane. For more details, see the `plugin usage guide
<https://pennylane-forest.readthedocs.io/en/latest/usage.html>`_ and refer to the PennyLane
documentation.


Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects
or applications built on PennyLane.


Authors
=======

PennyLane-Forest is the work of `many contributors <https://github.com/PennyLaneAI/pennylane-forest/graphs/contributors>`_.

If you are doing research using PennyLane and PennyLane-Forest, please cite `our paper <https://arxiv.org/abs/1811.04968>`_:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, M. Sohaib Alam, Shahnawaz Ahmed,
    Juan Miguel Arrazola, Carsten Blank, Alain Delgado, Soran Jahangiri, Keri McKiernan, Johannes Jakob Meyer,
    Zeyue Niu, Antal Száva, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. arXiv:1811.04968



Support
=======

- **Source Code:** https://github.com/PennyLaneAI/pennylane-forest
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane-forest/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove


License
=======

PennyLane-Forest is **free** and **open source**, released under the BSD 3-Clause `license
<https://github.com/PennyLaneAI/pennylane-forest/blob/master/LICENSE>`_.

.. license-end-inclusion-marker-do-not-remove