PennyLane-Forest Plugin
#######################

:Release: |release|

.. image:: _static/puzzle_forest.png
    :align: center
    :width: 70%
    :target: javascript:void(0);

|

.. include:: ../README.rst
  :start-after:	header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove

Once Pennylane-Forest is installed, the provided Forest devices can be accessed straight
away in PennyLane, without the need to import any additional packages.

Devices
~~~~~~~
Currently, PennyLane-Forest provides these Forest devices for PennyLane:

.. devicegalleryitem::
    :name: 'forest.numpy_wavefunction'
    :description: Forest's Numpy wavefunction simulator backend.
    :link: devices/numpy_wavefunction.html

.. devicegalleryitem::
    :name: 'forest.wavefunction'
    :description: The Forest SDK wavefunction simulator backend.
    :link: devices/wavefunction.html

.. devicegalleryitem::
    :name: 'forest.qvm'
    :description: Forest's QVM and pyQuil pyQVM simulator.
    :link: devices/qvm.html

.. raw:: html

        <div style='clear:both'></div>
        </br>


Tutorials
~~~~~~~~~

To see the PennyLane-Forest plugin in action, you can use any of the qubit based `demos
from the PennyLane documentation <https://pennylane.ai/qml/demonstrations.html>`_, for example
the tutorial on `qubit rotation <https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html>`_,
and simply replace ``'default.qubit'`` with a ``'forest.XXX'`` device:

.. code-block:: python

    dev = qml.device('forest.XXX', wires=XXX)

Tutorials that originally showcase the forest device are the demos on
`noisy devices <https://pennylane.ai/qml/demos/pytorch_noise.html>`_ and
`ensemble classification <https://pennylane.ai/qml/demos/tutorial_ensemble_multi_qpu.html>`_.

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

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code
