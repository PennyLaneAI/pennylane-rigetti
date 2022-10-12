QVM and quilc server configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    If using the downloadable Forest SDK with the default server configurations
    for the QVM and the Quil compiler (i.e., you launch them with the commands
    ``qvm -S`` and ``quilc -R``), then you will not need to set these keyword arguments.
    If using a non-default port or host for either of the servers, see the 
    `pyQuil configuration documentation <https://pyquil-docs.rigetti.com/en/stable/advanced_usage.html#pyquil-configuration>`_
    for details on how to override the default values.

    Likewise, if you are running PennyLane using the Rigetti Quantum Cloud Service (QCS)
    and have logged in with the 
    `QCS CLI <https://docs.rigetti.com/qcs/guides/using-the-qcs-cli#configuring-credentials>`__, 
    these environment variables are set automatically and will also not need to be passed in PennyLane.
