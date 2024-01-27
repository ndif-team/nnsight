.. raw:: html
   
   <script>

      const info = {
         "tutorial-card-0": {
            "title": "Walkthrough",
            "text": "Main Features",
         },
         "tutorial-card-1": {
            "title": "IOI Patching",
            "text": "arXiv:2211.00593",
         },
         "tutorial-card-2": {
            "title": "Attribution Patching",
            "text": "arXiv:2308.09124",
         },
         "tutorial-card-3": {
            "title": "Logit Lens",
            "text": "nostalgebraist",
         },
         "tutorial-card-4": {
            "title": "Future Lens",
            "text": "arXiv:2311.04897",
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


Tutorials
=========

.. card::
   :class-body: page-info 
   :shadow: none 

   Activation patching!

.. grid:: 4
   :class-container: tutorial-card-section

   .. grid-item::

      .. card::
         :img-top: images/particle_background.jpg
         :class-card: sd-text-black sd-border-0 tutorials-cards

         Walkthrough
   
   .. grid-item::

      .. card::
         :img-top: images/particle_background.jpg
         :class-card: sd-text-black sd-border-0 tutorials-cards
         
         Activation Patching

   .. grid-item::

      .. card::
         :img-top: images/particle_background.jpg
         :class-card: sd-text-black sd-border-0 tutorials-cards

         Attribution Patching
   
   .. grid-item::

      .. card::
         :img-top: images/particle_background.jpg
         :class-card: sd-text-black sd-border-0 tutorials-cards

         Logit Lens
   


.. .. grid:: 2
..    :gutter: 2

..    .. grid-item-card:: Walkthrough
..       :link: notebooks/walkthrough.ipynb

..       :bdg-primary:`Main Features`

..    .. grid-item-card:: IOI Patching
..       :link: notebooks/tutorials/ioi_patching.ipynb

..       :bdg-primary:`arXiv:2211.00593`

..    .. grid-item-card:: Attribution Patching
..       :link: notebooks/tutorials/attribution_patching.ipynb

..       :bdg-primary:`arXiv:2308.09124`

..    .. grid-item-card:: Logit Lens
..       :link: notebooks/tutorials/logit_lens.ipynb

..       :bdg-primary:`nostalgebraist`

..    .. grid-item-card:: Future Lens
..       :link: notebooks/tutorials/future_lens.ipynb

..       :bdg-primary:`arXiv:2311.04897`
   
..    .. grid-item-card:: Function Vectors
..       :link: notebooks/tutorials/function_vectors.ipynb

..       :bdg-primary:`arXiv:2310.15213`
   
..    .. grid-item-card:: Dictionary Learning
..       :link: notebooks/tutorials/sae.ipynb
   
..       :bdg-primary:`arXiv:2309.08600`


.. div:: hidden-toc

   .. toctree::
      :maxdepth: 1

      notebooks/tutorials/ioi_patching.ipynb
      notebooks/tutorials/attribution_patching.ipynb
      notebooks/tutorials/logit_lens.ipynb
      notebooks/tutorials/future_lens.ipynb
      notebooks/tutorials/function_vectors.ipynb
      notebooks/tutorials/sae.ipynb


