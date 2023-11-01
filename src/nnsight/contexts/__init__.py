"""The contexts module contains logic for managing the tracing and running of models with :mod:`nnsight.tracing` and :mod:`nnsight.module`

The primary two classes involved here are :class:`nnsight.contexts.Tracer.Tracer` and :class:`nnsight.contexts.Invoker.Invoker`.

The Tracer class creates a :class:`nnsight.tracing.Graph.Graph` around the meta_model of a :class:`nnsight.models.AbstractModel.AbstractModel` which tracks and manages the operations performed on the inputs and outputs of said model.
Modules in the meta_model expose their `.output` and `.input` attributes which when accessed, add to the computation graph of the tracer.
To do this they need to know about the current Tracer object so each Module's `.tracer` object is set to be the current Tracer.
The Tracer object also keeps track of the batch_size of the most recent input as well as the generation index for multi iteration generations.
It also keeps track of all of the inputs made during it's context in the `.batched_input` attribute. Inputs added to this attribute should be in a format where each index is a batch index and allows the model to batch all of the inputs together.
This is to keep things consistent where if two different inputs are in two different valid formats, they both become the same format and are easy to batch.
In the case of LanguageModels, regardless if the input are string prompts, pre-processed dictionaries, or input ids, the batched input is only input ids.
On exiting the Tracer context, the Tracer object should use the information and inputs provided to it to carry out the execution of the model and update its `.output` attribute with the result.

The Invoker class should be what actually accepts inputs to the model/graph, and updates it's parent Tracer object with the appropriate information about said input.
On entering the invoker context with some input, the invoker leverages the model to pre-process and prepare the input to the model.
Using the prepared inputs, it updates it's Tracer object with a batched version of the input, the size of the batched input, and the current generation idx.
It also runs a 'meta' version of the input through the model's meta_model. This updates the sizes/dtypes of all of the Module's inputs and outputs based on the characteristics of the input.

`nnsight` comes with two types of Tracers, `nnsight.contexts.Generator.Generator` and `nnsight.contexts.Runner.Runner`

The `nnsight.contexts.Generator.Generator` Tracer is the more feature rich context manager built for multi-iteration and multi-input executions of the model.
Each generator context uses invoker contexts within to add new inputs to the eventual execution like `generator.invoke(...)`
The Generator object also has logic to perform the execution of the traced computation graph on NDIF remote servers (assuming they are running!)

The `nnsight.contexts.Runner.Runner` Tracer is a simpler context manager that is both a Tracer as well as an Invoker. Meaning you pass input directly to the Runner object and on it's contextual exit, executes the model and computation graph.

"""