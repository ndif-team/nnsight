Features
=========

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

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 
      :link: notebooks/features/getting.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-wrench fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Getting</h5>
               <p class="card-text">Access values</p>
            </div>
         </div>

   .. grid-item-card::
      :link: notebooks/features/setting.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-code-pull-request fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Setting</h5>
               <p class="card-text">Intervene on values</p>
            </div>
         </div>

   .. grid-item-card::
      :link: notebooks/features/scan_validate.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-binoculars fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Scan and Validate</h5>
               <p class="card-text">Debug tensor shapes</p>
            </div>
         </div>


   .. grid-item-card:: 
      :link: notebooks/features/operations.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-glasses fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Operations</h5>
               <p class="card-text">Edit values</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/modules.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-cubes fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Modules</h5>
               <p class="card-text">Apply modules</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/custom_functions.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-atom fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Custom Functions</h5>
               <p class="card-text">Add thing to the Intervention Graph</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/gradients.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-backward fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Gradients</h5>
               <p class="card-text">Intervene on gradients</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/early_stopping.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-circle-stop fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Early Stopping</h5>
               <p class="card-text">Save computation time</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/conditionals.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-code-branch fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Conditional Interventions</h5>
               <p class="card-text">Use If Needed</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/cross_prompt.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-shuffle fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Cross Prompts</h5>
               <p class="card-text">Edit in one pass</p>
            </div>
         </div>
   
   .. grid-item-card:: 
      :link: notebooks/features/multiple_token.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-gears fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Generation</h5>
               <p class="card-text">Generate multiple tokens</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/model_editing.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-pen-to-square fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Model Editing</h5>
               <p class="card-text">Add persistent interventions</p>
            </div>
         </div>


   .. grid-item-card:: 
      :link: notebooks/features/remote_execution.ipynb
      :class-card: surface
      :class-body: surface

   
      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-satellite-dish fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Remote Execution</h5>
               <p class="card-text">Use our servers</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/sessions.ipynb
      :class-card: surface
      :class-body: surface

   
      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-bars fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Sessions</h5>
               <p class="card-text">Do many traces in one request</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/streaming.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-paper-plane fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Streaming</h5>
               <p class="card-text">Send remote values to local</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/iterator.ipynb
      :class-card: surface
      :class-body: surface

   
      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-arrow-rotate-left fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Iterative Interventions</h5>
               <p class="card-text">Make loops</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/lora_training.ipynb
      :class-card: surface
      :class-body: surface

   
      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-tower-broadcast fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">LORA</h5>
               <p class="card-text">Train one</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/vllm_support.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 15px; height: 100%;">
               <i class="fa-solid fa-stopwatch fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">vLLM Support</h5>
               <p class="card-text">Fast inference</p>
            </div>
         </div>
   

Report Issues
-------------

NNsight and NDIF are open-source and you can report issues, read, and clone the full source at https://github.com/ndif-team/nnsight. 
Also check out https://discuss.ndif.us/ to ask questions about our features or suggest new ones.

.. toctree::
   :glob:
   :maxdepth: 1
   
   notebooks/features/*

