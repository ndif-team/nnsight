About nnsight
==========

.. card:: How can you study the internals of a deep network too large for you to run?

    In the era of deep learning, the most interesting models for study require resources beyond the reach of many researchers. 
    The nnsight library, backed by remote services like NDIF, solves this problem by running “deep inference” on models which are hosted on shared servers.

.. figure:: _static/images/remote_execution.png

    An overview of the nnsight/NDIF pipeline. Researchers write simple, readable Python code specifying their experiments, and these experiments are run remotely on the NDIF server.

The nnsight library enables researchers to access the internals of large deep learned models. 
It uses simple python developer friendly syntax to provide direct access into the computation graph of models, and tracks operations you make on the inputs and outputs of modules to build an intervention graph. 
This intervention graph is then interleaved with that of the original model's computation graph locally, or remotely in conjunction with the NDIF (National Deep Inference Server)

What do you need to do?
------------------------

As a researcher, your experience with nnsight is that of just running experiment code on your local work station, just like you would run a normal pytorch experiment. In fact, the same code can be used to execute experiments locally on small models or to execute experiments remotely on large models, just by switching a few arguments.

The difference here is that when you use nnsight, the model internals are fully transparent to you enabling exponentially more avenues for experimentation.
Instead of running the model as a black box, you have access to model internals by reading or writing the activations at any layer of your network.
Configure your experiment by writing natural python code like the following:

.. code-block:: python
    :linenos:

    from nnsight import LanguageModel

    # Loading nnsight wrapped gpt2 model.
    model = LanguageModel('gpt2')

    # Enter intervention context to trace operations on the model and execute the resultant intervention graph remotely on NDIF
    with model.forward(remote=True) as runner:

        # Denote we wish to enter a prompt, and have traced operation within it's context be performed on it.
        with runner.invoke('The Eiffel Tower is in the city of ') as invoker:

            # Access the desired layer and ask for it's input to be stored and returned.
            hidden_states = model.transformer.h[1].input.save()

            # Set the output values of a subsequent layer's mlp module to zero.
            model.transformer.h[-1].mlp.output = 0

    # Upon exiting the context blocks, a request is made to NDIF containing the interventions we traced.
    # The output of the execution is populated in runner.output.
    output = runner.output
    # The value we asked for is injected into hidden_states.value. 
    hidden_states = hidden_states.value

The library is easy to use. Just import nnsight and like on line 4, you can instantiate any huggingface model by naming it using it's repo id. 
The first trick is on line 7. Instead of invoking and executing the model with a single call, we use a context via python with blocks to trace operations you perform, only to eventually run the model with those operations.
On line 10, an inner invocation context is created. This context now traces operations specific to the givin prompt.
Lines 13 and 16 demonstrate the getting and setting operations on module inputs and output.

How does it work?
------------------
Again, these operations are not executed immediately but instead adds to an intervention graph that is executed alongside the model's computation graph upon exit of the with block.
An example of one such intervention graph can be seen below:

.. figure:: _static/images/intrgraph.png

    An example of an intervention graph. Operations in research code create nodes in the graph which depend on module inputs and outputs as well as other nodes. Then, this intervention graph is interleaved with the normal computation graph of the chosen model, and requested inputs and outputs are injected into the intervention graph for execution. 
    

Building from these simple features, techniques such as causal mediation analysis be constructed and executed on models like Llama-2-70b!
In addition to getting and setting, nnsight gives full access to gradients and optimizations methods, out of order module applications, cross prompt interventions and much more.

See the :doc:`tutorials/notebooks/main_demo` and :doc:`tutorials/features` pages for more information on nnsight functionality.


The project is currently in Alpha pre-release and is looking for early users/and contributors!

Check out the `NDIF Discord <hhttps://discord.gg/ZRPgsf6P>`_ for updates, feature requests, bug reports and opportunities to help with the effort.



