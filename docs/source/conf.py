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
]

templates_path = ["_templates"]
exclude_patterns = []
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
    "navbar_end": ["theme-switcher","navbar-icon-links", "ndif_status"],
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/JadenFiotto-Kaufman/nnsight",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/6uFJmCSwW7",
            "icon": "fa-brands fa-discord",
        },
        {
            "name": "Status: Unknown",
            "url": "/status",
            "icon": "fa-solid fa-circle-check",
            "attributes": {"class": "ndif"},
        },
    ],
    "show_prev_next": False,
    "pygment_dark_style": "monokai",
    "pygment_light_style": "solarized-light"
}

html_js_files = [
    'js/custom.js',
    'js/code.js'
]