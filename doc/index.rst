PennyLane-Forest Plugin
#######################

:Release: |release|

.. include:: ../README.rst
  :start-after:	header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove

Once Pennylane-Forest is installed, the provided Forest devices can be accessed straight
away in PennyLane, without the need to import any additional packages.

Devices
~~~~~~~
Currently, PennyLane-Forest provides these Forest devices for PennyLane:

.. title-card::
    :name: 'forest.numpy_wavefunction'
    :description: pyQuil's Numpy wavefunction simulator backend.
    :link: devices/numpy_wavefunction.html

.. title-card::
    :name: 'forest.wavefunction'
    :description: The QCS wavefunction simulator backend.
    :link: devices/wavefunction.html

.. title-card::
    :name: 'forest.qvm'
    :description: QCS QVM and pyQuil pyQVM simulator.
    :link: devices/qvm.html

.. title-card::
    :name: 'forest.qpu'
    :description: QCS QPU.
    :link: devices/qpu.html

.. raw:: html

        <div style='clear:both'></div>
        </br>


Tutorials
~~~~~~~~~

Check out these demos to see the PennyLane-Forest plugin in action:

.. raw:: html

    <div class="row">

.. title-card::
    :name: Ensemble classification with QCS and Qiskit devices
    :description: Use two QPUs in parallel to help solve a machine learning classification problem.
    :link:  https://pennylane.ai/qml/demos/ensemble_multi_qpu.html

.. title-card::
    :name: PyTorch and noisy devices
    :description: Use PyTorch and a noisy QVM to see how optimization responds to noisy qubits.
    :link:  https://pennylane.ai/qml/demos/pytorch_noise.html

.. raw:: html

    </div></div><div style='clear:both'> <br/>

You can also try it out using any of the qubit based `demos from the PennyLane documentation
<https://pennylane.ai/qml/demonstrations.html>`_, for example the tutorial on
`qubit rotation <https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html>`_.
Simply replace ``'default.qubit'`` with a ``'forest.XXX'`` device if you have an API key for
hardware access.

.. code-block:: python

    dev = qml.device('forest.XXX', wires=XXX)

.. raw:: html

    <br/>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   installation
   support

.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   devices/numpy_wavefunction
   devices/wavefunction
   devices/qvm
   devices/qpu

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code
