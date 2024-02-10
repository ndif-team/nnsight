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
    newElement.style.height = '70vh';
    newElement.style.width = '100vw';
    newElement.style.top = '30vh';

    element.appendChild(newElement);

  </script>


.. raw:: html

    <section class="my-5">
      <div class="container col-xxl-8 px-4 py-5">
        <div class="row flex-lg-row-reverse d-flex justify-content-between g-5 py-5">
          <div class="col-10 col-sm-8 col-lg-7 mt-2">
            <img src="_static/images/nnsight_logo.svg" class="d-block mx-lg-auto img-fluid" alt="Bootstrap Themes" width="800" loading="lazy">
          </div>
          <div class="col-lg-5">
            <h1 class="display-5 fw-bold text-body-emphasis lh-1 mb-3">Transparent Science on Black Box AI</h1>
            <p class="lead" id="myElement">NNsight is a package for interpreting and manipulating the internals of models.</p>
            <div class="d-grid gap-2 d-md-flex justify-content-md-start">
              <button type="button" class="btn btn-primary btn-md px-2">Get Started</button>
              <button type="button" class="btn btn-outline-secondary btn-md px-2">Docs</button>
              <button type="button" class="btn btn-outline-secondary btn-md px-2">Tutorials</button>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="container py-5">
      <div class="row justify-content-center">
        <div class="col-3 align-items-center justify-content-end">
          <div class="list-group rounded-3 tab-background py-3 px-3" id="list-tab" role="tablist">
            <a class="list-group-item list-group-item-action rounded-3 py-3 active" id="list-home-list" data-bs-toggle="list" href="#list-home" role="tab" aria-controls="list-home-list">
              Outputs
              <span class="d-block small opacity-50">Simple model output</span>
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

        <div class="col-6 align-items-stretch">
          <div class="tab-content" id="nav-tabContent">
            <div class="tab-pane fade show active" id="list-home" role="tabpanel" aria-labelledby="list-home-list">
              <script src="https://gist.github.com/cadentj/94b68b5e27a92ea6c2882b3488dc21d8.js"></script>
            </div>
            <div class="tab-pane fade" id="list-profile" role="tabpanel" aria-labelledby="list-profile-list">
              <script src="https://gist.github.com/cadentj/94b68b5e27a92ea6c2882b3488dc21d8.js"></script>
            </div>
            <div class="tab-pane fade" id="list-messages" role="tabpanel" aria-labelledby="list-messages-list">
              <script src="https://gist.github.com/cadentj/94b68b5e27a92ea6c2882b3488dc21d8.js"></script>
            </div>
          </div>
        </div>
      </div>
    </section>
    

  
    <div class="px-4 py-5 my-5 text-center">
      <div class="col-lg-10 mx-auto">
        <p class="lead mb-4">Direct access to model internals, from one to one trillion parameters. Intervene on activations or gradients in transformers, diffusers, and any Torch model. Full access to gradients and optimizations methods, out of order module applications, cross prompt interventions, and many more features.</p>
        <div class="d-grid gap-2 d-sm-flex justify-content-sm-center">
          <pre class="my-0"><code id="foo">$ pip install nnsight</code><button class="btn" data-clipboard-target="#foo"><i class="bi bi-clipboard" onclick></button></i></pre><button type="button" class="btn btn-outline-secondary btn-md px-4">Get Started</button>
        </div>
      </div>
    </div>

    
    <section class="custom-section py-5 end-container">

      <div class="container py-5">
        <div class="row">
          <div class="col-md-8">
            <!-- Content for the left side -->
            <h2>Build and extend in real-time with CSS variables</h2>
            <p>Bootstrap 5 is evolving with each release to better utilize CSS variables for global theme styles...</p>
            <a href="#">Learn more about CSS variables</a>
          </div>
        </div>
        <div class="row mt-5">
          <div class="col-md-6 px-3">
            <h4>Declare a Simple Torch model</h4>
            <p>Let's start by creating a simple Torch model and tracing activations.</p>
            <!-- Content for the right side -->
            <script src="https://gist.github.com/cadentj/bca1c4366e7e14143ecf27990a0bfa45.js"></script>
          </div>
          <div class="col-md-6 px-3">
            <h4>Trace and Save Activations</h4>
            <p>Call <code>.save()</code> on activations.</p>
            <!-- Content for the right side -->
            <script src="https://gist.github.com/cadentj/62630994cd8c0110f9ed6a587d9605e0.js"></script>
          </div>
        </div>
      </div>

      <div class="container py-5">
        <div class="row">
          <div class="col-md-8">
            <!-- Content for the left side -->
            <h2>Build and extend in real-time with CSS variables</h2>
            <p>Bootstrap 5 is evolving with each release to better utilize CSS variables for global theme styles...</p>
            <a href="#">Learn more about CSS variables</a>
          </div>
        </div>
        <div class="row mt-5">
          <div class="col-md-6">
            <h4>Using CSS variables</h4>
            <p>Bootstrap 5 is evolving with each release to better utilize CSS variables for global theme styles...</p>
            <!-- Content for the right side -->
            <script src="https://gist.github.com/cadentj/94b68b5e27a92ea6c2882b3488dc21d8.js"></script>
          </div>
          <div class="col-md-6">
            <h4>Using CSS variables</h4>
            <p>Bootstrap 5 is evolving with each release to better utilize CSS variables for global theme styles...</p>
            <!-- Content for the right side -->
            <script src="https://gist.github.com/cadentj/94b68b5e27a92ea6c2882b3488dc21d8.js"></script>
          </div>
        </div>
      </div>
    </section>

