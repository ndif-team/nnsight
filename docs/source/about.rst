
About nnsight
=============

An API for transparent science on black-box AI
----------------------------------------------

.. card:: How can you study the internals of a deep network that is too large for you to run?

    In this era of large-scale deep learning, the most interesting AI models are massive black boxes
    that are hard to run. Ordinary commercial inference service APIs let you interact with huge
    models, but they do not let you see model internals.

    The nnsight library is different: it gives you full access to all the neural network internals.
    When used together with a remote service like the `National Deep Inference Facility <https://ndif.us/>`_ (NDIF),
    it lets you run experiments on huge open models easily, with full transparent access.
    The nnsight library is also terrific for studying smaller local models.

.. figure:: _static/images/remote_execution.png

    An overview of the nnsight/NDIF pipeline. Researchers write simple Python code to run along with the neural network locally or remotely. Unlike commercial inference, the experiment code can read or write any of the internal states of the neural networks being studied.  This code creates a computation graph that can be sent to the remote service and interleaved with the execution of the neural network.

How you use nnsight
-------------------

Nnsight is built on pytorch.

Running inference on a huge remote model with nnsight is very similar to running a neural network locally on your own workstation.  In fact, with nnsight, the same code for running experiments locally on small models can also be used on large models, just by changing a few arguments.

The difference between nnsight and normal inference is that when you use nnsight, you do not treat the model as an opaque black box.
Instead, you set up a python ``with`` context that enables you to get direct access to model internals while the neural network runs.
Here is how it looks:

.. code-block:: python
    :linenos:

    from nnsight import LanguageModel
    model = LanguageModel('meta-llama/Llama-2-70b-hf')
    with model.forward(remote=True) as runner:
        with runner.invoke('The Eiffel Tower is in the city of ') as invoker:
            hidden_state = model.layers[10].input[0].save()  # save one hidden state
            model.layers[11].mlp.output = 0  # change one MLP module output
    print('The model predicts', runner.output)
    print('The internal state was', hidden_state.value)

The library is easy to use. Any HuggingFace model can be loaded into a ``LanguageModel`` object, as you can see on line 2.  Notice we are loading a 70-billion parameter model, which is ordinarily pretty difficult to load on a regular workstation since it would take 140-280 gigabytes of GPU RAM just to store the parameters. 

The trick that lets us work with this huge model is on line 3.  We set the flag ``remote=True`` to indicate that we want to actually run the network on the remote service.  By default the remote service will be NDIF.  If we want to just run a smaller model quickly, we could leave it as ``remote=False``.

Then when we invoke the model on line 4, we do not just call it as a function. Instead, we use it as a ``with`` context manager.  The reason is that nnsight does not treat neural network models as black boxes; it provides direct access to model internals.

You can see what simple direct access looks like on lines 5-6.  On line 5, we grab a hidden state at layer 10, and on layer 6, we change the output of an MLP module inside the transformer at layer 11.

When you run this ``with``-block code on lines 5 and 6 on your local workstation, it actually creates a computation graph storing all the calculations you want to do.  When the outermost ``with`` block is completed, all the defined calculations are sent to the remote server and executed there.  Then when it's all done, the results can be accessed on your local workstation as shown on line 7 and 8.

What happens behind the scenes?
-------------------------------
When using nnsight, it is helpful to understand that the operations are not executed immediately but instead adds to an intervention graph that is executed alongside the model's computation graph upon exit of the with block.

An example of one such intervention graph can be seen below:

.. figure:: _static/images/intrgraph.png

    An example of an intervention graph. Operations in research code create nodes in the graph which depend on module inputs and outputs as well as other nodes. Then, this intervention graph is interleaved with the normal computation graph of the chosen model, and requested inputs and outputs are injected into the intervention graph for execution. 

Basic access to model internals can give you a lot of insight about what is going on inside a large model as it runs.  For example, you can use the `logit lens <https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens>`_ to read internal hidden states as text.  And use can use `causal tracing <https://rome.baulab.info/>`_ or `path patching <https://arxiv.org/abs/2304.05969>`_ or `other circuit discovery methods <https://arxiv.org/abs/2310.10348>`_ to locate the layers and components within the network that play a decisive role in making a decision.

And with nnsight, you can use these methods on large models like Llama-2-70b.

The nnsight library also provies full access to gradients and optimizations methods, out of order module applications, cross prompt interventions and many more features.

See the :doc:`start` and :doc:`features` pages for more information on nnsight functionality.

The project is currently in Alpha pre-release and is looking for early users/and contributors!

If you are interested in contributing or being an early user, join the `NDIF Discord <https://discord.gg/6uFJmCSwW7>`_ for updates, feature requests, bug reports and opportunities to help with the effort.
