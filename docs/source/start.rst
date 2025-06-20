Getting Started
===============

**NNsight** (/ɛn.saɪt/) is a package for the interpreting and manipulating the internals of deep learning models.

.. _installation:

Installation
------------

To get started with NNsight, install it with ``pip``. 

.. code-block:: console

   pip install nnsight

Please give the project a :ndif:`star on Github` to support the project. NNsight is open-source and you can read and clone the full source at https://github.com/ndif-team/nnsight.

Remote Model Access
-------------------

To remotely access LLMs through NDIF, you must sign up for an NDIF API key.

:bdg-link-primary:`NDIF API Key Registration <https://login.ndif.us/>`

NDIF hosts multiple LLMs, including various sizes of the Llama 3.1 models and DeepSeek-R1 models. 
All of our models are open for public use, but you need to apply for access to the Llama-3.1-405B models. 
You can view the full list of hosted models at https://nnsight.net/status/.

If you have a clear research need for Llama-3.1-405B and would like more details about applying for access, 
please refer to our  `405B pilot program application <https://ndif.us/405b.html>`_.

Access LLM Internals
--------------------

Now that you have your NDIF API key, you can start exploring LLM internals with NDIF and NNsight. 
We've put together a Colab notebook to help you get started.

:bdg-link-primary:`Open Colab <https://colab.research.google.com/github/ndif-team/ndif-website/blob/onboarding-fixes/public/notebooks/NDIFGetStarted.ipynb>`

This notebook will walk you through the following steps:

#. Installing NNsight
#. Setting up your NDIF API key
#. Loading a LLM in NNsight
#. Accessing and altering LLM internals remotely


Next Steps
-----------

.. grid:: 2 2 2 2 
   :gutter: 2

   .. grid-item-card:: Walkthrough
      :link: notebooks/tutorials/walkthrough.ipynb

      Walk through the basic functionality of the package.

   .. grid-item-card:: Remote Access
      :link: notebooks/features/remote_execution.ipynb

      Configure API access for remote model execution.

   .. grid-item-card:: Features
      :link: features
      :link-type: doc

      Check out the basic features provided by :bdg-primary:`nnsight`.

   .. grid-item-card:: Tutorials
      :link: tutorials
      :link-type: doc

      See :bdg-primary:`nnsight` implementations of common interpretability techniques.

   .. grid-item-card:: Forum
      :link: https://discuss.ndif.us/

      Discuss :bdg-primary:`nnsight`, NDIF, and more!



