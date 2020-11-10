pennylane_forest.QVMDevice
==========================

.. currentmodule:: pennylane_forest

.. autoclass:: QVMDevice
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

      ~QVMDevice.author
      ~QVMDevice.cache
      ~QVMDevice.circuit_hash
      ~QVMDevice.compiled_program
      ~QVMDevice.name
      ~QVMDevice.num_executions
      ~QVMDevice.obs_queue
      ~QVMDevice.observables
      ~QVMDevice.op_queue
      ~QVMDevice.operations
      ~QVMDevice.parameters
      ~QVMDevice.pennylane_requires
      ~QVMDevice.program
      ~QVMDevice.short_name
      ~QVMDevice.shots
      ~QVMDevice.state
      ~QVMDevice.version
      ~QVMDevice.wire_map
      ~QVMDevice.wires

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

      ~QVMDevice.access_state
      ~QVMDevice.active_wires
      ~QVMDevice.analytic_probability
      ~QVMDevice.apply
      ~QVMDevice.apply_parametric_program
      ~QVMDevice.apply_rotations
      ~QVMDevice.batch_execute
      ~QVMDevice.capabilities
      ~QVMDevice.check_validity
      ~QVMDevice.define_wire_map
      ~QVMDevice.estimate_probability
      ~QVMDevice.execute
      ~QVMDevice.execution_context
      ~QVMDevice.expval
      ~QVMDevice.generate_basis_states
      ~QVMDevice.generate_samples
      ~QVMDevice.map_wires
      ~QVMDevice.marginal_prob
      ~QVMDevice.mat_vec_product
      ~QVMDevice.post_apply
      ~QVMDevice.post_measure
      ~QVMDevice.pre_apply
      ~QVMDevice.pre_measure
      ~QVMDevice.probability
      ~QVMDevice.reset
      ~QVMDevice.sample
      ~QVMDevice.sample_basis_states
      ~QVMDevice.states_to_binary
      ~QVMDevice.statistics
      ~QVMDevice.supports_observable
      ~QVMDevice.supports_operation
      ~QVMDevice.var

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
