# Configuration file for the Sphinx documentation builder.

# Project Information
project = 'nnsight'
copyright = '2023, NDIF'
author = 'Jaden Fiotto-Kaufman'
release = '0.0.7'



# General Configuration
extensions = [
    'sphinx.ext.autodoc', # Auto documentation from docstrings
    'sphinx.ext.napoleon', # Support for NumPy and Google style docstrings
    'sphinx_copybutton', # Copy button for code blocks
    'sphinx_design', # Boostrap design components
    'nbsphinx', # Jupyter notebook support
]

templates_path = ['_templates']
exclude_patterns = []
fixed_sidebar = True



# HTML Output Options

# See https://sphinx-themes.org/ for more
html_theme = "pydata_sphinx_theme"

html_static_path = ['_static']

html_favicon = '_static/images/icon.ico'

html_theme_options = {
  "logo": {"text":"nnsight"},
  "show_nav_level": 2,
  "navbar_end": ["navbar-icon-links", "ndif_status"],
  "navbar_align": "left",
  "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/JadenFiotto-Kaufman/nnsight",
            "icon": "fa-brands fa-github",
        },
  ]
}

html_context = {
   "default_mode": "light",
   "ndif_url" : "https://ndif.baulab.us/ping"
}

html_css_files = [
    'css/custom.css',
]

