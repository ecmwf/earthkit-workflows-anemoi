# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import os
import sys

read_the_docs_build = os.environ.get("READTHEDOCS", None) == "True"

sys.path.insert(0, os.path.join(os.path.abspath(".."), "src"))


source_suffix = ".rst"
master_doc = "index"
pygments_style = "sphinx"
html_theme_options = {"logo_only": True}
html_logo = "_static/logo.png"


# -- Project information -----------------------------------------------------

project = "earthkit-workflows-anemoi"

author = "ECMWF"

year = datetime.datetime.now().year
if year == 2024:
    years = "2024"
else:
    years = "2024-%s" % (year,)

copyright = "%s, ECMWF" % (years,)


try:
    from anemoi.inference._version import __version__

    release = __version__
except ImportError:
    release = "0.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxarg.ext",
    "sphinxcontrib.autodoc_pydantic",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "'**.ipynb_checkpoints'"]

intersphinx_mapping = {
    "python": ("https://python.readthedocs.io/en/latest", None),
    "anemoi-utils": (
        "https://anemoi-utils.readthedocs.io/en/latest/",
        ("../../anemoi-utils/docs/_build/html/objects.inv", None),
    ),
    "anemoi-datasets": (
        "https://anemoi-datasets.readthedocs.io/en/latest/",
        ("../../anemoi-datasets/docs/_build/html/objects.inv", None),
    ),
    "anemoi-inference": (
        "https://anemoi-inference.readthedocs.io/en/latest/",
        ("../../anemoi-inference/docs/_build/html/objects.inv", None),
    ),
}


# https://www.notion.so/Deepnote-Launch-Buttons-63c642a5e875463495ed2341e83a4b2a


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Make the page width match the browser width
html_theme_options = {
    "body_max_width": "none",
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["style.css"]


todo_include_todos = not read_the_docs_build

autodoc_member_order = "bysource"  # Keep file order


# https://autodoc-pydantic.readthedocs.io/en/stable/users/configuration.html

autodoc_pydantic_model_show_json = True
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_member_order = "bysource"

html_context = {
    "display_github": True,
    "github_user": "ecmwf",
    "github_repo": "earthkit-workflows-anemoi",
    "github_version": "main/docs/",
}
