PennyLane-Rigetti Plugin
#######################

.. warning:: 
    The PennyLane-Rigetti plugin is only compatible with PennyLane v0.40 or below. To use Rigetti hardware with newer versions of PennyLane please use the `PennyLane-Braket plugin <https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/stable/index.html>`__ instead.

:Release: |release|

.. include:: ../README.rst
  :start-after:	header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove

Once Pennylane-Rigetti is installed, the provided Rigetti devices can be accessed straight
away in PennyLane, without the need to import any additional packages.

Devices
~~~~~~~
Currently, PennyLane-Rigetti provides these Rigetti devices for PennyLane:

.. title-card::
    :name: 'rigetti.numpy_wavefunction'
    :description: pyQuil's Numpy wavefunction simulator backend.
    :link: devices/numpy_wavefunction.html

.. title-card::
    :name: 'rigetti.wavefunction'
    :description: The QCS wavefunction simulator backend.
    :link: devices/wavefunction.html

.. title-card::
    :name: 'rigetti.qvm'
    :description: QCS QVM and pyQuil pyQVM simulator.
    :link: devices/qvm.html

.. title-card::
    :name: 'rigetti.qpu'
    :description: QCS QPU.
    :link: devices/qpu.html

.. raw:: html

        <div style='clear:both'></div>
        </br>


Tutorials
~~~~~~~~~

Check out these demos to see the PennyLane-Rigetti plugin in action:

.. raw:: html

    <div class="row">

.. title-card::
    :name: Ensemble classification with QCS and Qiskit devices
    :description: <img src="https://pennylane.ai/_images/ensemble_diagram.png" width="100%" />
    :link:  https://pennylane.ai/qml/demos/ensemble_multi_qpu.html

.. title-card::
    :name: PyTorch and noisy devices
    :description: <img src="https://pennylane.ai/_images/bloch.gif" width="100%" />
    :link:  https://pennylane.ai/qml/demos/pytorch_noise.html

.. raw:: html

    </div></div><div style='clear:both'> <br/>

You can also try it out using any of the qubit based `demos from the PennyLane documentation
<https://pennylane.ai/qml/demonstrations.html>`_, for example the tutorial on
`qubit rotation <https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html>`_.
Simply replace ``'default.qubit'`` with a ``'rigetti.XXX'`` device if you have an API key for
hardware access.

.. code-block:: python

    dev = qml.device('rigetti.XXX', wires=XXX)

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
