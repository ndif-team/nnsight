.. role:: raw-html(raw)
   :format: html

.. raw:: html
   
   <script>

      const info = {
         "tutorial-card-0": {
            "title": "Walkthrough",
            "text": "Learn how to use the central components of the library.",
         },
         "tutorial-card-1": {
            "title": "Activation Patching",
            "text": "Using causal interventions to identify which activations in a model matter for producing some output.",
         },
         "tutorial-card-2": {
            "title": "Attribution Patching",
            "text": "Using gradients to take a linear approximation of activation patching.",
         },
         "tutorial-card-3": {
            "title": "Logit Lens",
            "text": "Decoding activations into vocabulary space.",
         },
         "tutorial-card-4": {
            "title": "Future Lens",
            "text": "Anticipating subsequent tokens from a single hidden state.",
         },
         "tutorial-card-5": {
            "title": "Function Vectors",
            "text": "How do language models represent functions?",
         },
         "tutorial-card-6": {
            "title": "Dictionary Learning",
            "text": "Finding highly interpretable features in langauge models.",
         },
      }

      window.onload = function() {
         var tutorialCards = document.getElementsByClassName('tutorials-cards');
         for (var i = 0; i < tutorialCards.length; i++) {
            // Assign a unique ID to each card
            tutorialCards[i].id = 'tutorial-card-' + i;

            // Add a click event listener to each card
            tutorialCards[i].addEventListener('click', function() {
               var title = info[this.id]['title'];
               var newText = info[this.id]['text'];
               changeText(title, newText);
            });
         }

         function changeText(title, newText) {
            Array.from(document.getElementsByClassName("page-info")).forEach((elm) => {
                  Array.from(elm.getElementsByClassName('sd-card-text')).forEach((text) => {
                     text.innerHTML = '<b>' + title + '</b>' + '<br>' + newText;
                  });
            });
         }
      }
   </script>

   <style>
      .toctree-wrapper {
         display: none !important;
      }
   </style>


Tutorials
=========

.. card::
   :class-body: page-info 
   :shadow: none 

   **Click for more information**
   :raw-html:`<br />` 

.. grid:: 2 2 3 4
   :class-container: tutorial-card-section

   .. grid-item::

      .. card::
         :img-top: _static/images/tutorials/walkthrough.png
         :class-card: sd-text-black sd-border-0 tutorials-cards
         :class-body: sd-font-weight-bold

         `Walkthrough <notebooks/tutorials/walkthrough.ipynb>`_
   
   .. grid-item::

      .. card::
         :img-top: _static/images/tutorials/activation_patching.png
         :class-card: sd-text-black sd-border-0 tutorials-cards
         :class-body: sd-font-weight-bold
         
         `Activation Patching <notebooks/tutorials/ioi_patching.ipynb>`_

   .. grid-item::

      .. card::
         :img-top: _static/images/tutorials/attribution_patching.png
         :class-card: sd-text-black sd-border-0 tutorials-cards
         :class-body: sd-font-weight-bold

         `Attribution Patching <notebooks/tutorials/attribution_patching.ipynb>`_
   
   .. grid-item::

      .. card::
         :img-top: _static/images/tutorials/logit_lens.png
         :class-card: sd-text-black sd-border-0 tutorials-cards
         :class-body: sd-font-weight-bold

         `Logit Lens <notebooks/tutorials/logit_lens.ipynb>`_

   .. grid-item::

      .. card::
         :img-top: _static/images/tutorials/future_lens.png
         :class-card: sd-text-black sd-border-0 tutorials-cards
         :class-body: sd-font-weight-bold

         `Future Lens <notebooks/tutorials/future_lens.ipynb>`_

   .. grid-item::

      .. card::
         :img-top: _static/images/tutorials/function_vectors.png
         :class-card: sd-text-black sd-border-0 tutorials-cards
         :class-body: sd-font-weight-bold

         `Function Vectors <notebooks/tutorials/function_vectors.ipynb>`_

   .. grid-item::

      .. card::
         :img-top: _static/images/tutorials/dictionary_learning.png
         :class-card: sd-text-black sd-border-0 tutorials-cards
         :class-body: sd-font-weight-bold

         `Dictionary Learning <notebooks/tutorials/sae.ipynb>`_


.. toctree::
   :maxdepth: 1

   notebooks/tutorials/walkthrough.ipynb
   notebooks/tutorials/ioi_patching.ipynb
   notebooks/tutorials/attribution_patching.ipynb
   notebooks/tutorials/logit_lens.ipynb
   notebooks/tutorials/future_lens.ipynb
   notebooks/tutorials/function_vectors.ipynb
   notebooks/tutorials/sae.ipynb


