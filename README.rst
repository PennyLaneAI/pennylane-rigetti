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

.. image:: https://readthedocs.com/projects/xanaduai-pennylane-forest/badge/?version=latest&style=flat-square
    :alt: Read the Docs
    :target: https://docs.pennylane.ai/projects/forest

.. image:: https://img.shields.io/pypi/v/pennylane-forest.svg?style=flat-square
    :alt: PyPI
    :target: https://pypi.org/project/pennylane-forest

.. image:: https://img.shields.io/pypi/pyversions/pennylane-forest.svg?style=flat-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/pennylane-forest

.. header-start-inclusion-marker-do-not-remove

The PennyLane Forest plugin allows different Rigetti devices to work with
PennyLane --- the wavefunction simulator, the Quantum Virtual Machine (QVM), and Quantum Processing Units (QPUs).

`pyQuil <https://pyquil.readthedocs.io>`__ is a Python library for quantum programming using the
quantum instruction language (Quil) --- resulting quantum programs can be executed using the
`Rigetti Forest SDK <https://pyquil-docs.rigetti.com/en/stable/>`__ and `Rigetti Quantum Cloud Services (QCS)
<https://qcs.rigetti.com/>`__.

`PennyLane <https://pennylane.readthedocs.io>`__ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.


.. header-end-inclusion-marker-do-not-remove

The plugin documentation can be found here: `<https://docs.pennylane.ai/projects/forest>`__.

Features
========

* Provides four devices to be used with PennyLane: ``forest.numpy_wavefunction``,
  ``forest.wavefunction``, ``forest.qvm``, and ``forest.qpu``. These provide access to the pyQVM
  Numpy wavefunction simulator, pyQuil wavefunction simulator, quantum
  virtual machine (QVM), and quantum processing units (QPUs) respectively.


* All provided devices support all core qubit PennyLane operations and observables.

.. installation-start-inclusion-marker-do-not-remove

Installation
============

PennyLane-Forest, as well as all required Python packages mentioned above, can be installed via ``pip``:
::

   	$ python -m pip install pennylane-forest


Make sure you are using the Python 3 version of pip.

Alternatively, you can install PennyLane-Forest from the source code by navigating to the top-level directory and running
::

	$ python setup.py install

Dependencies
~~~~~~~~~~~~

PennyLane-Forest requires the following libraries be installed:

* `Python <http://python.org/>`__ >=3.7

as well as the following Python packages:

* `PennyLane <http://pennylane.readthedocs.io/>`__ >=0.15.0
* `pyQuil <https://pyquil-docs.rigetti.com/en/stable/>`__ >=3.3.1, <4.0.0

If you currently do not have Python 3 installed, we recommend
`Anaconda for Python 3 <https://www.anaconda.com/download/>`__, a distributed version
of Python packaged for scientific computation.

Additionally, if you would like to compile the quantum instruction language (Quil) and run it
locally using a quantum virtual machine (QVM) server, you will need to download and install the
Forest software development kit (SDK):

* `Forest SDK <https://pyquil-docs.rigetti.com/en/stable/>`__

Alternatively, you may sign up for Rigetti's Quantum Cloud Services (QCS)  which will allow you to compile your 
quantum code and run on real QPUs. Note that this requires a valid QCS account and the QCS CLI:

* `QCS <https://docs.rigetti.com/en/>`__
* `QCS CLI <https://docs.rigetti.com/qcs/guides/using-the-qcs-cli>`__

Tests
~~~~~

To test that the PennyLane-Forest plugin is working correctly you can run

.. code-block:: bash

    $ make test

in the source folder.

Documentation
~~~~~~~~~~~~~

To build the HTML documentation, go to the top-level directory and run:

.. code-block:: bash

  $ make docs


The documentation can then be found in the ``doc/_build/html/`` directory.

.. installation-end-inclusion-marker-do-not-remove

Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`__ containing your contribution.
All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects
or applications built on PennyLane.


Authors
=======

PennyLane-Forest is the work of `many contributors <https://github.com/PennyLaneAI/pennylane-forest/graphs/contributors>`__.

If you are doing research using PennyLane and PennyLane-Forest, please cite `our paper <https://arxiv.org/abs/1811.04968>`__:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, M. Sohaib Alam, Shahnawaz Ahmed,
    Juan Miguel Arrazola, Carsten Blank, Alain Delgado, Soran Jahangiri, Keri McKiernan, Johannes Jakob Meyer,
    Zeyue Niu, Antal Száva, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. arXiv:1811.04968

.. support-start-inclusion-marker-do-not-remove

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
<https://github.com/PennyLaneAI/pennylane-forest/blob/master/LICENSE>`__.

.. license-end-inclusion-marker-do-not-remove
