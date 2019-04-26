PennyLane-Forest
################

:Release: |release|
:Date: |today|



This PennyLane plugin allows the Rigetti Forest and pyQuil simulators to be used as PennyLane devices.

`pyQuil <https://pyquil.readthedocs.io>`_ is a Python library for quantum programming using the quantum instruction language (Quil) - resulting quantum programs can be executed using the Rigetti Forest platform.

`PennyLane <https://pennylane.readthedocs.io>`_ is a machine learning library for optimization and automatic differentiation of hybrid quantum-classical computations.


Features
========

* Provides four devices to be used with PennyLane: ``forest.numpy_wavefunction``, ``forest.wavefunction``, ``forest.qvm``, and ``forest.qpu``. These provide access to the pyQVM Numpy wavefunction simulator, Forest wavefunction simulator, quantum virtual machine (QVM), and quantum processing unit (QPU) respectively.


* All provided devices support all core qubit PennyLane operations and expectation values.


* Provides custom PennyLane operations to cover additional pyQuil operations: ``T``, ``S``, ``ISWAP``, ``CCNOT``, ``PSWAP``, and many more. Every custom operation supports analytic differentiation.


* Leverage PennyLane’s automatic differentiation and optimization together with Rigetti's Forest SDK and Quantum Cloud Services.


To get started with the PennyLane Strawberry Fields plugin, follow the :ref:`installation steps <installation>`, then see the :ref:`usage <usage>` page.

Authors
=======

`Josh Izaac <https://github.com/josh146>`_, `Keri A. McKiernan <https://github.com/kmckiern>`_

Contents
========

.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installing
   usage

.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 1
   :caption: Code details

   code/ops
   code/device
   code/numpy_wavefunction
   code/wavefunction
   code/qvm
   code/qpu
