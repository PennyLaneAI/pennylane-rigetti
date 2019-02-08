.. _installation:

Installation and setup
######################


Dependencies
============

.. highlight:: bash

PennyLane-Forest requires the following libraries be installed:

* `Python <http://python.org/>`_ >=3.6

as well as the following Python packages:

* `PennyLane <http://pennylane.readthedocs.io/>`_
* `pyQuil <http://docs.rigetti.com/>`_

If you currently do not have Python 3 installed, we recommend `Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed version of Python packaged for scientific computation.

Additionally, if you would like to compile the quantum instruction language (Quil) and run it locally using a quantum virtual machine (QVM) server, you will need to download and install the Forest software development kit (SDK):

* `Forest SDK <https://www.rigetti.com/forest>`_

Alternatively, you may sign up for Rigetti's Quantum Cloud Services (QCS) to acquire a Quantum Machine Image (QMI) which will allow you to compile your quantum code and run on real quantum processing units (QPUs), or on a preinstalled QVM. Note that this requires a valid QCS account.

* `Quantum Cloud Services <https://www.rigetti.com/>`_

Installation
============

Installation of PennyLane-Forest, as well as all required Python packages mentioned above, can be installed via ``pip``:
::

   	$ python -m pip install pennylane-forest


Make sure you are using the Python 3 version of pip.

Alternatively, you can install PennyLane-Forest from the source code by navigating to the top directory and running
::

	$ python setup.py install


Software tests
==============

To ensure that PennyLane-Forest is working correctly after installation, the test suite can be run by navigating to the source code folder and running
::

	$ make test


Documentation
=============

To build the HTML documentation, go to the top-level directory and run
::

  $ make docs

The documentation can then be found in the :file:`doc/_build/html/` directory.
