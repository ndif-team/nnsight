# Installation

Run `pip install -r requirements.txt` while in the `nnsight/docs` directory.

Optionally, run `pip install -e nnsight` from the root directory to use the working dir

If you haven't already, you should also install pandoc. With brew run: `brew install pandoc`

# Adding Tutorials

Tutorials are written as Jupyter Notebooks in `.ipynb` format, and automatically converted to html by the `nbsphinx` extension.

The only requirement of `nbsphinx` is that you add a header to the notebook to act as the rendered title on Sphinx docs. To do this, create a markdown cell at the top of your notebook with a header `#`.

Then, just add your notebook to the `nnsight/docs/source/notebooks/tutorials` directory.

If you're adding a new notebook, navigate to `nnsight/docs/source/tutorials.rst`. At the bottom of the page, add the path to your notebook under the `toctree`.

```
.. toctree::
   :maxdepth: 1

   notebooks/tutorials/walkthrough.ipynb
   ...
   <YOUR NOTEBOOK PATH>

```

# Compiling Sphinx to HTML

Run `make dirhtml` from the `nnsight/docs` directory. The build is located in `nnsight/docs/build/dirhtml`. You can run `python3 -m http.server <PORT>` from that directory, and see the site at `http://localhost:<PORT>`.
