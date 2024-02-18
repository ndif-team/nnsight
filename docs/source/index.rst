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

.. raw:: html

  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/default.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>

  <!-- and it's easy to individually load additional languages -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>

  <script>hljs.highlightAll();</script>

  <script>
    document.addEventListener('DOMContentLoaded', (event) => {
      document.querySelectorAll('img.transparent').forEach(el => {
        el.style.background = 'transparent';
      });
    });
  </script>

.. raw:: html

    <script type="text/javascript">
    document.addEventListener('DOMContentLoaded', function() {
        var observer = new MutationObserver(function(mutations) {
            updateCSSLink();
        });
        observer.observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
        
        function updateCSSLink() {
            const dark = document.documentElement.dataset.theme == 'dark';
            var link = document.querySelector('link[rel="stylesheet"][href*="hljs"]');
            if(link) {
                link.href = dark ? '../_static/css/hljs-dark.css?v=0.2' : '../_static/css/hljs-light.css?v=0.2';
            }
        }
        
        // Initial check in case the theme is already set before the observer is added
        updateCSSLink();
    });
    </script>
    <link rel="stylesheet" href="../_static/css/hljs-dark.css?v=0.2">


.. raw:: html

    <section class="d-flex align-items-center" style="height: 85vh;">
        <div class="container">
            <div class="row g-5">
                <div class="col-md-6">
                    <h1 class="display-5 fw-bold lh-1 mb-3 text">Interpretable Neural Networks</h1>
                    <p class="lead text" id="myElement">NNsight (/ɛn.saɪt/) is a package for interpreting and manipulating the internals of models.</p>
                    <div class="d-grid gap-2 d-md-flex">
                        <a href="start" role="button" class="btn btn-primary primary btn-md px-4">Start</a>
                        <a href="documentation" role="button" class="btn btn-outline-secondary secondary btn-md px-4">Docs</a>
                        <a href="features" role="button" class="btn btn-outline-secondary btn-md secondary px-4">Tutorials</a>
                        <a href="about" role="button" class="btn btn-outline-secondary btn-md secondary px-4">About</a>
                    </div>
                </div>
                <div class="col-md-6 mt-2 d-none d-md-block">
                    <img src="_static/images/nnsight_logo.svg" class="img-fluid transparent" alt="Bootstrap Themes" loading="lazy">
                </div>
            </div>
        </div>
    </section>

    <section class="d-flex align-items-center mid-section d-none d-md-block">
      <div class="px-4 text-center">
        <div class="col-lg-10 mx-auto">
          <p class="lead mb-4">Direct access to model internals, from one to one hundred billion parameters. Intervene on activations or gradients in transformers, train optimizations methods, perform out of order module applications, cross prompt interventions, and so much more.</p>
        </div>
      </div>
    </section>

    <section class="d-flex align-items-center d-none d-md-block" style="height: 60vh;">
        <div class="container">
            <div class="row align-items-end pb-3 border-bottom mb-5">
                <div class="col">
                    <h2 class="mb-0">Features</h2>
                </div>
                <div class="col text-end">
                    <a href="features" style="text-decoration:none">See More →</a>
                </div>
            </div>
            <div class="row g-3">
                <div class="col-md-4">
                    <div class="list-group rounded-3 tab-background p-3" id="list-tab" role="tablist">
                        <a class="list-group-item list-group-item-action rounded-3 py-3 active" id="list-home-list" data-bs-toggle="list" href="#list-home" role="tab">
                            Wrap any PyTorch model
                            <span class="d-block d-none d-md-block small opacity-50">NNsight class object</span>
                        </a>
                        <a class="list-group-item list-group-item-action rounded-3 py-3" id="list-profile-list" data-bs-toggle="list" href="#list-profile" role="tab">
                            Access any hidden state
                            <span class="d-block d-none d-md-block small opacity-50">Expose Module inputs and outputs </span>
                        </a>
                        <a class="list-group-item list-group-item-action rounded-3 py-3" id="list-messages-list" data-bs-toggle="list" href="#list-messages" role="tab">
                            Develop complex interventions
                            <span class="d-block d-none d-md-block small opacity-50">Edit module outputs and measure effect</span>
                        </a>
                    </div>
                </div>
                <div class="col-md-8">
                    <div class="tab-content static-on-small" id="nav-tabContent">
                        <div class="tab-pane fill-height fade show active" id="list-home" role="tabpanel" aria-labelledby="list-home-list">
                            <pre class="code-surface fill-height"><code class="language-python fill-height simple"></code></pre>
                        </div>
                        <div class="tab-pane fill-height fade" id="list-profile" role="tabpanel" aria-labelledby="list-profile-list">
                            <pre class="code-surface fill-height"><code class="language-python fill-height trace"></code></pre>
                        </div>
                        <div class="tab-pane fill-height fade" id="list-messages" role="tabpanel" aria-labelledby="list-messages-list">
                            <pre class="code-surface fill-height"><code class="language-python fill-height multi"></code></pre>
                        </div>
                    </div>
                </div>
                
            </div>
        </div>
    </section>

    <section class="d-none d-md-block"  style="height: 20vh;"/>


.. raw:: html
    
    <script>

        var element = document.querySelector('.bd-container');

        if (element) {
        // If .bd-container exists, proceed to create the new element
        var newElement = document.createElement('div');
        newElement.id = 'canvas';

        newElement.style.position = 'fixed';
        newElement.style.zIndex = '-1';
        newElement.style.height = '100vh';
        newElement.style.width = '100vw';
        newElement.style.top = '0';

        element.appendChild(newElement);
        } else {
            console.log(".bd-container element does not exist.");
        }
    </script>