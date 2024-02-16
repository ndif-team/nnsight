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
      h5 {
         margin-top: 0 !important;
      }
   </style>


Tutorials
=========

.. grid:: 2 2 2 2
   :class-container: tutorial-card-section
   :gutter: 3

   .. grid-item-card:: 
      :link: notebooks/tutorials/walkthrough.ipynb
      :class-card: code-surface
      :class-body: code-surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <img src="../_static/images/tutorials/walkthrough.png" class="img-fluid" style="max-width: 50px; margin-right: 20px;">
            <div>
               <h5 class="code-surface card-title">Walkthrough</h5>
               <p class="code-surface card-text">Learn the basics</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/ioi_patching.ipynb
      :class-card: code-surface
      :class-body: code-surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <img src="../_static/images/tutorials/activation_patching.png" class="img-fluid" style="max-width: 50px; margin-right: 20px;">
            <div>
               <h5 class="code-surface card-title">Activation Patching</h5>
               <p class="code-surface card-text">Causal intervention</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/attribution_patching.ipynb
      :class-card: code-surface
      :class-body: code-surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <img src="../_static/images/tutorials/attribution_patching.png" class="img-fluid" style="max-width: 50px; margin-right: 20px;">
            <div>
               <h5 class="code-surface card-title">Attribution Patching</h5>
               <p class="code-surface card-text">Approximate patching</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/logit_lens.ipynb
      :class-card: code-surface
      :class-body: code-surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <img src="../_static/images/tutorials/logit_lens.png" class="img-fluid" style="max-width: 50px; margin-right: 20px;">
            <div>
               <h5 class="code-surface card-title">Logit Lens</h5>
               <p class="code-surface card-text">Decode activations</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/future_lens.ipynb
      :class-card: code-surface
      :class-body: code-surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <img src="../_static/images/tutorials/future_lens.png" class="img-fluid" style="max-width: 50px; margin-right: 20px;">
            <div>
               <h5 class="code-surface card-title">Future Lens</h5>
               <p class="code-surface card-text">Probe future tokens</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/function_vectors.ipynb
      :class-card: code-surface
      :class-body: code-surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <img src="../_static/images/tutorials/function_vectors.png" class="img-fluid" style="max-width: 50px; margin-right: 20px;">
            <div>
               <h5 class="code-surface card-title">Function Vectors</h5>
               <p class="code-surface card-text">Lambdas</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/sae.ipynb
      :class-card: code-surface
      :class-body: code-surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <img src="../_static/images/tutorials/dictionary_learning.png" class="img-fluid" style="max-width: 50px; margin-right: 20px;">
            <div>
               <h5 class="code-surface card-title">Dictionary Learning</h5>
               <p class="code-surface card-text">Sparse autoencoders</p>
            </div>
         </div>



.. toctree::
   :maxdepth: 1

   notebooks/tutorials/walkthrough.ipynb
   notebooks/tutorials/ioi_patching.ipynb
   notebooks/tutorials/attribution_patching.ipynb
   notebooks/tutorials/logit_lens.ipynb
   notebooks/tutorials/future_lens.ipynb
   notebooks/tutorials/function_vectors.ipynb
   notebooks/tutorials/sae.ipynb


