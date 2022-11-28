# Release 0.28.0-dev

### New features since last release

### Breaking changes

### Improvements

* With the introduction of custom measurement classes, all the `MeasurementProcess.return_type`
  checks have been changed by `isinstance` checks.
  [(#388)](https://github.com/PennyLaneAI/pennylane-lightning/pull/388)

### Documentation

### Bug fixes

### Contributors

This release contains contributions from (in alphabetical order):

Albert Mitjans-Coma
---

# Release 0.27.0

### New features since last release

#### Re-introduction of the Rigetti Quantum Proccessing Unit (QPU) device

This release uses the latest version of [pyQuil](https://github.com/rigetti/pyquil)
to connect to [Rigetti Quantum Cloud Services (QCS)](https://docs.rigetti.com/qcs/)
and enables the use of the latest [Rigetti QPUs](https://qcs.rigetti.com/qpus)
as a PennyLane device.
[#107](https://github.com/PennyLaneAI/pennylane-forest/pull/107)

### Breaking changes

* The package has been renamed to `pennylane-rigetti`. The top level import is now
  `pennylane_rigetti`. In addition, the prefix for device short names have been
  changed to `rigetti`. For example, `rigetti.qpu`.
  [#110](https://github.com/PennyLaneAI/pennylane-forest/pull/110)

* A new version of the QCS CLI is required if you want to use your QCS account
  to run your workloads on a live Rigetti QPU. See 
  [Using the QCS CLI](https://docs.rigetti.com/qcs/guides/using-the-qcs-cli) for
  details.
  [#107](https://github.com/PennyLaneAI/pennylane-forest/pull/107)

* The `forest_url` parameter has been removed, as it is now managed by the QCS CLI.
  [#107](https://github.com/PennyLaneAI/pennylane-forest/pull/107)

* The `compiler_url` and `qvm_url` device parameters have been removed. The default
  URLs can be overridden using the `QCS_SETTINGS_APPLICATIONS_PYQUIL_QUILC_URL`
  and `QCS_SETTINGS_APPLICATIONS_PYQUIL_QVM_URL` environment variables,
  respectively.
  [#107](https://github.com/PennyLaneAI/pennylane-forest/pull/107)

* The `timeout` parameter for all devices has been renamed to
  `compiler_timeout`.
  [#107](https://github.com/PennyLaneAI/pennylane-forest/pull/107)

* There is now a default execution timeout of 10 seconds. This can be configured
  for a device by using the new `execution_timeout` parameter.
  [#107](https://github.com/PennyLaneAI/pennylane-forest/pull/107)

* The `S`, `T`, `CSWAP`, `ISWAP`, and `CCNOT` operations have been removed.
  Import them directly from `pennylane` instead.
  [#107](https://github.com/PennyLaneAI/pennylane-forest/pull/107)

### Improvements

* Improves the computation of the expectation value when using `QPUDevice` by
  skipping the `Device.generate_samples` method.
  [#108](https://github.com/PennyLaneAI/pennylane-forest/pull/108)

### Bug fixes

* The QPU device now correctly sets the number of shots when parametric 
  compilation is disabled. 
  [#107](https://github.com/PennyLaneAI/pennylane-forest/pull/107)

### Contributors

This release contains contributions from (in alphabetical order):

Albert Mitjans Coma, Antal Szava, Marquess Valdez.

---

# Release 0.24.0

### Bug fixes

* Defines the missing `state` method and `returns_state` entry of the
  `capabilities` dictionary for `forest.wavefunction` and
  `forest.numpy_wavefunction`.
  [(#101)](https://github.com/PennyLaneAI/pennylane-forest/pull/101)

### Contributors

This release contains contributions from (in alphabetical order):

Antal Száva.

---

# Release 0.20.0

### Bug fixes

* Fix a bug where array parameters where not accepted when building circuits on
  pyQuil side.
  [(#90)](https://github.com/PennyLaneAI/pennylane-forest/pull/90)

### Improvements

* Added support for Python 3.10.
  [(#96)](https://github.com/PennyLaneAI/pennylane-forest/pull/96)

### Contributors

This release contains contributions from (in alphabetical order):

Romain Moyard.

---

# Release 0.17.0

### Breaking changes

* Deprecated Python 3.6.
  [(#85)](https://github.com/PennyLaneAI/pennylane-forest/pull/85)

### Improvements

* Removed a validation check for ``QubitUnitary`` that is now in PennyLane
  core.
  [(#74)](https://github.com/PennyLaneAI/pennylane-forest/pull/74)

### Bug fixes

* Pins the PyQuil version to use as `pyquil>=2.16,<2.28.3` due to API
  deprecations in PyQuil version 3.0.
  [(#73)](https://github.com/PennyLaneAI/pennylane-forest/pull/77)

### Contributors

This release contains contributions from (in alphabetical order):

Theodor Isacsson, Romain Moyard, Antal Száva.

---

# Release 0.16.0

### Bug fixes

* Fixed a bug caused by the `expand_state` method always assuming that

  inactive wires are the least significant bits.
  [(#73)](https://github.com/PennyLaneAI/pennylane-forest/pull/73)

### Contributors

This release contains contributions from (in alphabetical order):

Antal Száva.

---

# Release 0.15.0

### Breaking changes

* For compatibility with PennyLane v0.15, the `analytic` keyword argument
  has been removed from all devices. Analytic expectation values can
  still be computed by setting `shots=None`.
  [(#71)](https://github.com/PennyLaneAI/pennylane-forest/pull/71)

* For compatibility with PennyLane v0.15, parametric compilation now depends on
  the `requires_grad` attribute of parameters instead of the deprecated
  `Variable` class.
  [(#71)](https://github.com/PennyLaneAI/pennylane-forest/pull/71)

* The circuit hashes used for parametric compilation are computed in
  `QVMDevice` instead of in `QubitDevice` defined in Pennylane.
  [(#71)](https://github.com/PennyLaneAI/pennylane-forest/pull/71)

### Contributors

This release contains contributions from (in alphabetical order):

Antal Száva.

---

# Release 0.14.0

### New features since last release

### Improvements

* Updated the CI.
  [(#63)](https://github.com/PennyLaneAI/pennylane-forest/pull/63)

### Bug fixes

* Updated the plugin to be compatible with the new core of PennyLane.
  [(#67)](https://github.com/PennyLaneAI/pennylane-forest/pull/67)
  [(#68)](https://github.com/PennyLaneAI/pennylane-forest/pull/68)

### Documentation

* Adapted the documentation to the PennyLane theme.
  [(#64)](https://github.com/PennyLaneAI/pennylane-forest/pull/64)

### Contributors

This release contains contributions from (in alphabetical order):

Thomas Bromley, Theodor Isacsson, Josh Izaac, Maria Schuld, Antal Száva.

---

# Release 0.12.0

* Version bump

---

# Release 0.11.0

### New features

* Forest devices now support custom wire labels. #55 #57 #58 #59

  One can now specify any string or number as a custom wire label, and use these labels to address subsystems on the device:

  ```python
  dev = qml.device('forest.qvm' device='4q-qvm', wires=['q1', 'ancilla', 0, 1])

  @qml.qnode(dev)
  def circuit():
     qml.Hadamard(wires='q1')
     qml.CNOT(wires=[1, 'ancilla'])
     return qml.expval(qml.PauliZ(0))
  ```

### Bug fixes

* Fixes a bug where basis states were not correctly initialized #56

* Tensor measurements have been fixed due to a change in PL v0.11 #52

* The QCS no longer supports lattice-based devices such as Aspen-4-5Q-E. As a result, the `forest.qvm` and `forest.qpu` devices no longer accept lattice-based devices, and must be loaded with the full device name (e.g. Aspen-4). #51

### Contributors

Theodor Isacsson, Josh Izaac, Maria Schuld, Antal Száva

---

# Release 0.9.0

* Adding support for multi-qubit observable estimation.

* Added a new method `analytic_probability()` to the `Wavefunction` simulator devices, so that analytic and non-analytic probabilities are correctly returned.

* Reduced the default number of shots for the `QPUDevice` to 1000 from 1024.

* The test suite now programmatically queries `pyquil.list_quantum_computers()` for valid QPU lattices for integration tests, and additional gradient integration tests have been added.

* The minimum required version of PennyLane has been increased to v0.9 due to plugin API changes in PennyLane.

---

# Release 0.8.0

* Added parametric compilation for `forest.qvm` and `forest.qpu`, leading to significant speed increases when performing optimization

* The plugin has been ported to use the new [`QubitDevice`](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.QubitDevice.html) API for plugins, resulting in a cleaner interface with built-in support for multi-qubit observables.

* Bug fixes to allow the plugin to work with pyQuil >= 2.16

* Operator estimation on `forest.qpu` now works with both 1 and 2 qubit gates

* Adds a converter so that pyQuil and Quil programs can be loaded as differentiable PennyLane templates

---

# Release 0.6.0

* Added ability to return multi-qubit expectation values to all devices

* Added variance support

* Replaced dense matrix expansion with a tensordot implementation

* Added support for `samples`

* Readout Error Mitigation using Operator Estimation

* Updates the plugin to work with PyQuil 2.13 and PennyLane 0.6

---

# Release 0.3.0

### New features

* Updated to support PyQuil v2.9.0 and PennyLane v0.4.0. Note that PennyLane-Forest now requires PennyLane>=0.4

* The observable `qml.Hermitian(A, wires)` can now be measured on an arbitrary number of wires. That is, you can now pass a matrix A of size `[2**N, 2**N]` which acts on `N` qubits (i.e., `len(wires)==N`).

* Adds support for the `qml.var()` measurement return type provided by PennyLane v0.4.

---

# Release 0.2.0

* Using any of the available lattices at any given time, instead of a specifically named one

* updated minimum pyquil version requirement to 2.7

* added support for pyQVM

* removed 2.4 changes

* Updating compiler urls

* Replacing old lattice with new

* pyqvm working without the compiler

* Updating compiler urls

* Replacing old lattice with new

* added build and documentation badges

* Remove the LocalQVMCompiler from the tests
