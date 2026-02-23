# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import warnings

# Make the package importable without installing it
sys.path.insert(0, os.path.abspath("../src"))

# Suppress noisy third-party import-time warnings that aren't Sphinx issues
warnings.filterwarnings("ignore", category=UserWarning, module="dust_extinction")

# -- Project information -----------------------------------------------------

project = "fiesta"
copyright = "2024, Thibeau Wouters, Hauke Koehn"
author = "Thibeau Wouters, Hauke Koehn"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

# MyST extensions for Markdown files
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]

# Mock optional dependencies and flowMC submodules not present in all installs
autodoc_mock_imports = [
    "afterglowpy",
    "flowMC.Sampler",
    "flowMC.resource_strategy_bundle",
    "flowMC.resource_strategy_bundle.RQSpline_MALA",
]

# Autodoc defaults
autodoc_default_options = {
    "undoc-members": True,
    "show-inheritance": True,
    "members": True,
}

# Type hints rendering
typehints_defaults = "comma"
always_document_param_types = False

# Napoleon: support Google and NumPy docstring styles
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/nuclear-multimessenger-astronomy/fiestaEM",
    "use_repository_button": True,
    "show_toc_level": 2,
    "use_fullscreen_button": False,
    "use_download_button": False,
}

html_logo = "fiesta_logo.jpeg"
html_static_path = ["_static"]
html_css_files = ["style.css"]

# -- Copy button -------------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Suppress specific warnings ----------------------------------------------

suppress_warnings = [
    "ref.python",
]
