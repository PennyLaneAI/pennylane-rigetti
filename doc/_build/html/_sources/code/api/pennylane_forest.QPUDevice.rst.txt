pennylane_forest.QPUDevice
==========================

.. currentmodule:: pennylane_forest

.. autoclass:: QPUDevice
   :show-inheritance:

   .. raw:: html

      <a class="attr-details-header collapse-header" data-toggle="collapse" href="#attrDetails" aria-expanded="false" aria-controls="attrDetails">
         <h2 style="font-size: 24px;">
            <i class="fas fa-angle-down rotate" style="float: right;"></i> Attributes
         </h2>
      </a>
      <div class="collapse" id="attrDetails">

   .. autosummary::
      :nosignatures:

      ~QPUDevice.author
      ~QPUDevice.cache
      ~QPUDevice.circuit_hash
      ~QPUDevice.compiled_program
      ~QPUDevice.name
      ~QPUDevice.num_executions
      ~QPUDevice.obs_queue
      ~QPUDevice.observables
      ~QPUDevice.op_queue
      ~QPUDevice.operations
      ~QPUDevice.parameters
      ~QPUDevice.pennylane_requires
      ~QPUDevice.program
      ~QPUDevice.short_name
      ~QPUDevice.shots
      ~QPUDevice.state
      ~QPUDevice.version
      ~QPUDevice.wire_map
      ~QPUDevice.wires

   .. autoattribute:: author
   .. autoattribute:: cache
   .. autoattribute:: circuit_hash
   .. autoattribute:: compiled_program
   .. autoattribute:: name
   .. autoattribute:: num_executions
   .. autoattribute:: obs_queue
   .. autoattribute:: observables
   .. autoattribute:: op_queue
   .. autoattribute:: operations
   .. autoattribute:: parameters
   .. autoattribute:: pennylane_requires
   .. autoattribute:: program
   .. autoattribute:: short_name
   .. autoattribute:: shots
   .. autoattribute:: state
   .. autoattribute:: version
   .. autoattribute:: wire_map
   .. autoattribute:: wires

   .. raw:: html

      </div>

   .. raw:: html

      <a class="meth-details-header collapse-header" data-toggle="collapse" href="#methDetails" aria-expanded="false" aria-controls="methDetails">
         <h2 style="font-size: 24px;">
            <i class="fas fa-angle-down rotate" style="float: right;"></i> Methods
         </h2>
      </a>
      <div class="collapse" id="methDetails">

   .. autosummary::

      ~QPUDevice.access_state
      ~QPUDevice.active_wires
      ~QPUDevice.analytic_probability
      ~QPUDevice.apply
      ~QPUDevice.apply_parametric_program
      ~QPUDevice.apply_rotations
      ~QPUDevice.batch_execute
      ~QPUDevice.capabilities
      ~QPUDevice.check_validity
      ~QPUDevice.define_wire_map
      ~QPUDevice.estimate_probability
      ~QPUDevice.execute
      ~QPUDevice.execution_context
      ~QPUDevice.expval
      ~QPUDevice.generate_basis_states
      ~QPUDevice.generate_samples
      ~QPUDevice.map_wires
      ~QPUDevice.marginal_prob
      ~QPUDevice.mat_vec_product
      ~QPUDevice.post_apply
      ~QPUDevice.post_measure
      ~QPUDevice.pre_apply
      ~QPUDevice.pre_measure
      ~QPUDevice.probability
      ~QPUDevice.reset
      ~QPUDevice.sample
      ~QPUDevice.sample_basis_states
      ~QPUDevice.states_to_binary
      ~QPUDevice.statistics
      ~QPUDevice.supports_observable
      ~QPUDevice.supports_operation
      ~QPUDevice.var

   .. automethod:: access_state
   .. automethod:: active_wires
   .. automethod:: analytic_probability
   .. automethod:: apply
   .. automethod:: apply_parametric_program
   .. automethod:: apply_rotations
   .. automethod:: batch_execute
   .. automethod:: capabilities
   .. automethod:: check_validity
   .. automethod:: define_wire_map
   .. automethod:: estimate_probability
   .. automethod:: execute
   .. automethod:: execution_context
   .. automethod:: expval
   .. automethod:: generate_basis_states
   .. automethod:: generate_samples
   .. automethod:: map_wires
   .. automethod:: marginal_prob
   .. automethod:: mat_vec_product
   .. automethod:: post_apply
   .. automethod:: post_measure
   .. automethod:: pre_apply
   .. automethod:: pre_measure
   .. automethod:: probability
   .. automethod:: reset
   .. automethod:: sample
   .. automethod:: sample_basis_states
   .. automethod:: states_to_binary
   .. automethod:: statistics
   .. automethod:: supports_observable
   .. automethod:: supports_operation
   .. automethod:: var

   .. raw:: html

      </div>

   .. raw:: html

      <script type="text/javascript">
         $(".collapse-header").click(function () {
             $(this).children('h2').eq(0).children('i').eq(0).toggleClass("up");
         })
      </script>
