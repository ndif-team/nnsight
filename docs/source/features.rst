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
   </style>

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 
      :link: notebooks/features/getting.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-wrench fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Getting</h5>
               <p class="card-text">Accessing values</p>
            </div>
         </div>

   .. grid-item-card::
      :link: notebooks/features/setting.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-code-pull-request fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Setting</h5>
               <p class="card-text">Intervening on values</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/operations.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-glasses fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Operations</h5>
               <p class="card-text">Editing values</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/modules.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-cubes fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Modules</h5>
               <p class="card-text">Applying modules</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/cross_prompt.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-shuffle fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Cross Prompt Intervention</h5>
               <p class="card-text">Edit in one pass</p>
            </div>
         </div>
   
   .. grid-item-card:: 
      :link: notebooks/features/multiple_token.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-gears fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Generation</h5>
               <p class="card-text">Multiple tokens</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/gradients.ipynb
      :class-card: surface
      :class-body: surface


      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-backward fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Working With Gradients</h5>
               <p class="card-text">Intervene on gradients</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/remote_execution.ipynb
      :class-card: surface
      :class-body: surface

   
      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-satellite-dish fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Remote Execution</h5>
               <p class="card-text">Transparent deep learning</p>
            </div>
         </div>

   .. grid-item-card:: 
      :link: notebooks/features/token_indexing.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-regular fa-object-ungroup fa-2x"></i>
            </div> 
            <div>
               <h5 class="card-title">Multi-Token Indexing</h5>
               <p class="card-text">Easy token indexing</p>
            </div>
         </div>
   


.. toctree::
   :maxdepth: 1
   
   notebooks/features/getting.ipynb
   notebooks/features/setting.ipynb
   notebooks/features/operations.ipynb
   notebooks/features/modules.ipynb
   notebooks/features/cross_prompt.ipynb
   notebooks/features/multiple_token.ipynb
   notebooks/features/gradients.ipynb
   notebooks/features/remote_execution.ipynb
   notebooks/features/token_indexing.ipynb

