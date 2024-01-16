:html_theme.sidebar_secondary.remove:
:sd_hide_title:

nnsight
=======

.. toctree::
   :maxdepth: 1
   :hidden:

   start
   documentation
   features
   tutorials
   About <about>

.. grid:: 1 1 2 2
    :class-container: hero
    :reverse:

    .. grid-item:: 
        .. div:: 

          .. image:: _static/images/nnsight_logo.svg
            :width: 400
            :height: 276
            
    .. grid-item:: 

        .. div:: sd-fs-1 sd-font-weight-bold title-bot sd-text-primary image-container

            NNsight

        .. div:: sd-fs-4 sd-font-weight-bold sd-my-0 sub-bot image-container

            interpretable neural networks

        **NNsight** (/ɛn.saɪt/) is a package for interpreting and manipulating the internals of large models

        .. div:: button-group
        
          .. button-ref:: start
              :color: primary
              :shadow:

                  Get Started

          .. button-ref:: tutorials
            :color: primary
            :outline:
            
                Tutorials

          .. button-ref:: documentation
            :color: primary
            :outline:
            
                Docs


.. div:: sd-fs-1 sd-font-weight-bold sd-text-center sd-text-primary sd-mb-5

  Key Features

.. grid:: 1 1 2 2
    :class-container: features

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/one.png
          :width: 250

        .. div:: 

          **Integration**

          Pass in a ID from any HuggingFace Transformer repo and access its weights with nnsight. 

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/two.png
          :width: 250

        .. div:: 

          **Interpretability**

          Access the internal gradients and activations at any point or module in a model.
    
    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/two.png
          :width: 250

        .. div:: 

          **Intuitive**

          Set up a context block and manipulate model internals with only a couple lines of code.

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/one.png
          :width: 250

        .. div:: 

          **Interoperable**

          Enable grad and train interventions like LORA or probes on any point in a model.

