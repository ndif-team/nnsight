import time

# Configuration file for the Sphinx documentation builder.

# Project Information
project = "nnsight"
copyright = "2024 NDIF"
author = "Jaden Fiotto-Kaufman"


# General Configuration
extensions = [
    "sphinx.ext.autodoc",  # Auto documentation from docstrings
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx_copybutton",  # Copy button for code blocks
    "sphinx_design",  # Boostrap design components
    "nbsphinx",  # Jupyter notebook support
    "sphinx.ext.viewcode",  # Add source links to the generated HTML files
    "sphinx.ext.extlinks",  # Add external links
]

templates_path = ["_templates"]
fixed_sidebar = True

# HTML Output Options

# See https://sphinx-themes.org/ for more
html_theme = "pydata_sphinx_theme"
html_title = "nnsight"
html_logo = "_static/images/nnsight_logo.svg"
html_static_path = ["_static"]
html_show_sphinx = False

html_favicon = "_static/images/icon.ico"
html_show_sourcelink = False

html_context = {
   "default_mode": "dark",
   "ndif_url": "https://ndif.dev/ping",
   "version_identifier": str(int(time.time())),
}


html_theme_options = {
    "show_nav_level": 2,
    "navbar_end": ["ndif_status", "theme-switcher","navbar-icon-links"],
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "Status: Unknown",
            "url": "/status",
            "icon": "fa-solid fa-circle-check",
            "attributes": {"class": "ndif"},
        },
        {
            "name": "GitHub",
            "url": "https://github.com/ndif-team/nnsight",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discord",
            "url": "https://forms.gle/1Y6myaXYzSh3oHf56",
            "icon": "fa-brands fa-discord",
        },
    ],
    "show_prev_next": False,
    "pygment_dark_style": "monokai",
}

extlinks = {'ndif': ('https://%s.com/ndif-team/nnsight',
                      '%s')}

html_js_files = [
    'js/custom.js',
    'js/code.js'
]