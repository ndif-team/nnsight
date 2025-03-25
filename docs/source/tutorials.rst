.. role:: raw-html(raw)
   :format: html

.. raw:: html

   <script>
   document.addEventListener('DOMContentLoaded', (event) => {
      document.querySelectorAll('h5.card-title').forEach(el => {
      el.style.margin = '0';
      });
   });
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

.. grid:: 1 1 2 2
   :class-container: tutorial-card-section
   :gutter: 3

   .. grid-item-card:: 
      :link: notebooks/tutorials/walkthrough.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-person-walking fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Walkthrough</h5>
               <p class="card-text">Learn the basics</p>
            </div>
         </div>


   .. grid-item-card:: 
      :link: notebooks/tutorials/start_remote_access.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-satellite-dish fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Access LLMs</h5>
               <p class="card-text">Use our hosted models</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/activation_patching.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-code-pull-request fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Activation Patching</h5>
               <p class="card-text">Causal intervention</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/attribution_patching.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-diagram-project fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Attribution Patching</h5>
               <p class="card-text">Approximate patching</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/boundless_DAS.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-magnifying-glass fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Boundless DAS</h5>
               <p class="card-text">Identifying Causal Mechanisms in Alpaca</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/dict_learning.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-book-open fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Dictionary Learning</h5>
               <p class="card-text">Sparse autoencoders</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/logit_lens.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-arrow-down-a-z fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Logit Lens</h5>
               <p class="card-text">Decode activations</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/tutorials/LoRA_tutorial.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-sliders fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">LoRA</h5>
               <p class="card-text">Fine tuning for sentiment analysis</p>
            </div>
         </div>         

.. toctree::
   :glob:
   :maxdepth: 1

   notebooks/tutorials/*


