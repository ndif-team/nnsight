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
      document.querySelectorAll('code.language-python').forEach(el => {
        el.innerText = "def forward(self, x):";
      });
      document.querySelectorAll('img.transparent').forEach(el => {
        el.style.background = 'transparent';
      });
    });
  </script>

.. raw:: html
  
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
  <link rel="stylesheet" href="_static/css/front.css">
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

    <section class="hero">
      <div class="container px-4" style="margin-top: 20%;">
        <div class="row d-flex justify-content-between g-5" style="min-height: 100vh;">
          <div class="col-5">
            <h1 class="display-5 fw-bold text-body-emphasis lh-1 mb-3 text">Interpretable Neural Networks</h1>
            <p class="lead text" id="myElement">NNsight is a package for interpreting and manipulating the internals of models.</p>
            <div class="d-grid gap-2 d-md-flex justify-content-md-start">
              <button type="button" class="btn btn-primary primary btn-md px-2">Get Started</button>
              <button type="button" class="btn btn-outline-secondary btn-md px-2">Docs</button>
              <button type="button" class="btn btn-outline-secondary btn-md px-2">Tutorials</button>
            </div>
          </div>
          <div class="col-7 mt-2">
            <img src="_static/images/nnsight_logo.svg" class="d-block mx-lg-auto img-fluid transparent" alt="Bootstrap Themes" width="800" loading="lazy">
          </div>
        </div>
      </div>
    </section>


    
    <div class="px-4 text-center">
      <div class="col-lg-10 mx-auto">
        <p class="lead mb-4">Direct access to model internals, from one to one trillion parameters. Intervene on activations or gradients in transformers, diffusers, and any Torch model. Full access to gradients and optimizations methods, out of order module applications, cross prompt interventions, and many more features.</p>
        <div class="d-grid gap-2 d-sm-flex justify-content-sm-center">
          <pre class="my-0 surface"><code>pip install nnsight</code></pre><button type="button" class="btn btn-outline-secondary btn-md px-4">Get Started</button>
        </div>
      </div>
    </div>

    <section class="container py-5">
      <div class="row justify-content-center">
        <div class="col-3 align-items-center justify-content-end">
          <div class="list-group rounded-3 tab-background py-3 px-3" id="list-tab" role="tablist">
            <a class="list-group-item list-group-item-action rounded-3 py-3 active" id="list-home-list" data-bs-toggle="list" href="#list-home" role="tab" aria-controls="list-home-list">
              Outputs
              <span class="d-block small opacity-50">Simple model calls</span>
            </a>
            <a class="list-group-item list-group-item-action rounded-3 py-3 " id="list-profile-list" data-bs-toggle="list" href="#list-profile" role="tab" aria-controls="list-profile-list">
              Tracing
              <span class="d-block small opacity-50">Operate on the forward</span>
            </a>
            <a class="list-group-item list-group-item-action rounded-3 py-3" id="list-messages-list" data-bs-toggle="list" href="#list-messages" role="tab" aria-controls="list-messages-list">
              Generation
              <span class="d-block small opacity-50">Generate new tokens</span>
            </a>
          </div>
        </div>

        <div class="col-8 align-items-stretch">
          <div class="tab-content" id="nav-tabContent">
            <div class="tab-pane fade show active" id="list-home" role="tabpanel" aria-labelledby="list-home-list">
              <pre class="surface"><code class="language-python"></code></pre>
            </div>
            <div class="tab-pane fade" id="list-profile" role="tabpanel" aria-labelledby="list-profile-list">
              <pre class="surface"><code class="language-python"></code></pre>
            </div>
            <div class="tab-pane fade" id="list-messages" role="tabpanel" aria-labelledby="list-messages-list">
              <pre class="surface"><code class="language-python"></code></pre>
            </div>
          </div>
        </div>
      </div>
    </section>
