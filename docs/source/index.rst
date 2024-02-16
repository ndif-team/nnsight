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
  
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">

  <script src="dist/clipboard.min.js"></script>
  
  <script>
    new ClipboardJS('.btn');
  </script>

  <script>
      // Select the element with the class 'some-class'
    var element = document.querySelector('.bd-container');

    var newElement = document.createElement('div');
    newElement.id = 'canvas';

    newElement.style.position = 'fixed';
    newElement.style.zIndex = '-1';
    newElement.style.height = '100vh';
    newElement.style.width = '100vw';
    newElement.style.top = '0';

    element.appendChild(newElement);

  </script>


.. raw:: html

    <section class="d-flex align-items-center" style="height: 85vh;">
        <div class="container">
            <div class="row g-5">
                <div class="col-md-6">
                    <h1 class="display-5 fw-bold lh-1 mb-3 text">Interpretable Neural Networks</h1>
                    <p class="lead text" id="myElement">NNsight (/ɛn.saɪt/) is a package for interpreting and manipulating the internals of models.</p>
                    <div class="d-grid gap-2 d-md-flex">
                        <a href="start" role="button" class="btn btn-primary primary btn-md px-4">Get Started</a>
                        <a href="documentation" role="button" class="btn btn-outline-secondary secondary btn-md px-4">Docs</a>
                        <a href="features" role="button" class="btn btn-outline-secondary btn-md secondary px-4">Tutorials</a>
                    </div>
                </div>
                <div class="col-md-6 mt-2 d-none d-md-block">
                    <img src="_static/images/nnsight_logo.svg" class="img-fluid transparent" alt="Bootstrap Themes" loading="lazy">
                </div>
            </div>
        </div>
    </section>

    <section class="d-flex align-items-center" style="height: 50vh;">
      <div class="px-4 text-center">
        <div class="col-lg-10 mx-auto">
          <p class="lead mb-4">Direct access to model internals, from one to one trillion parameters. Intervene on activations or gradients in transformers, train optimizations methods, perform out of order module applications, cross prompt interventions, and so much more.</p>
          <div class="d-grid gap-2 d-sm-flex justify-content-sm-center">
            <pre class="my-0 surface"><code>pip install nnsight</code></pre><a type="button" href="start" role="button" class="btn btn-outline-secondary secondary btn-md px-4 center-text">Get Started</a>
          </div>
        </div>
      </div>
    </section>

    <section class="d-flex align-items-center" style="height: 60vh;">
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
                <div class="col-md-8 order-md-last">
                    <div class="tab-content" id="nav-tabContent">
                        <div class="tab-pane fill-height fade show active" id="list-home" role="tabpanel" aria-labelledby="list-home-list">
                            <pre class="code-surface fill-height"><code class="language-python fill-height simple"></code></pre>
                        </div>
                        <div class="tab-pane fill-height fade" id="list-profile" role="tabpanel" aria-labelledby="list-profile-list">
                            <pre class="code-surface fill-height"><code class="language-python fill-height simple"></code></pre>
                        </div>
                        <div class="tab-pane fill-height fade" id="list-messages" role="tabpanel" aria-labelledby="list-messages-list">
                            <pre class="code-surface fill-height"><code class="language-python fill-height simple"></code></pre>
                        </div>
                    </div>
                </div>
                <div class="col-md-4 order-md-first">
                    <div class="list-group rounded-3 tab-background p-3" id="list-tab" role="tablist">
                        <a class="list-group-item list-group-item-action rounded-3 py-3 active" id="list-home-list" data-bs-toggle="list" href="#list-home" role="tab">
                            A Simple Model Output
                            <span class="d-block d-none d-md-block small opacity-50">Simple model call</span>
                        </a>
                        <a class="list-group-item list-group-item-action rounded-3 py-3" id="list-profile-list" data-bs-toggle="list" href="#list-profile" role="tab">
                            Trace Activations and Gradients
                            <span class="d-block d-none d-md-block small opacity-50">Operate on the forward</span>
                        </a>
                        <a class="list-group-item list-group-item-action rounded-3 py-3" id="list-messages-list" data-bs-toggle="list" href="#list-messages" role="tab">
                            Edit at Every Generation
                            <span class="d-block d-none d-md-block small opacity-50">Generate new tokens</span>
                        </a>
                    </div>
                </div>
                
            </div>
        </div>
    </section>

    <section style="height: 20vh;"/>
