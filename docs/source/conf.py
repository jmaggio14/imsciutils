# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import sphinx_bootstrap_theme

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('.')) # add current dir as a module so we grab the sytle class

# load in version variables into 'version_info' dict
CURRENT_DIR = os.path.dirname(__file__)
VERSION_FILE = os.path.join(CURRENT_DIR,'..','..','imagepypelines','version_info.py')
with open(VERSION_FILE,'r') as f:
    raw = f.read()
version_info = {}
exec(raw,{},version_info)


# fetch the git branch sha for a permalink
import subprocess


# reference the pygments style defined in style.py
pygments_style = "style.ImagePypelinesLight"
# pygments_style = "style.ImagePypelinesDark"

# see Sphinx: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-pygments_style
################################################################################
# # JM: GET THE GIT SHA FOR PERMALINKS (this should be replaced down the line with tagged releases)
# # for travis builds
# if 'TRAVIS' in os.environ:
#     # for pull requests
#     if os.environ['TRAVIS_PULL_REQUEST_SHA'] != "":
#         GIT_SHA = os.environ['TRAVIS_PULL_REQUEST_SHA']
#     else:
#         GIT_SHA = os.environ['TRAVIS_COMMIT']
# # for local builds
# else:
#     GIT_SHA = subprocess.check_output("git rev-parse --verify HEAD").decode('utf-8').replace(' ','')


# -- Project information -----------------------------------------------------
project = "ImagePypelines"
copyright = version_info['__copyright__'].replace("Copyright (c)","")
author = version_info['__author__']


# The short X.Y version
version = '.'.join(version_info["__version__"].split('.',2)[:2])
# The full version, including alpha/beta/rc tags
release = version_info["__version__"]


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

def setup(app):
    # adds a copybutton for our code examples
    app.add_javascript('js/copybutton.js')
    # bootswatch theme
    # app.add_stylesheet('css/bootstrap.min.css')
    # our custom additions
    app.add_stylesheet('css/ip_custom.css')

# JM idk what this does but the automodapi docs say you should do this
numpydoc_show_class_members = False

extensions = [
    'sphinx_gallery.gen_gallery', # JM: turns .py examples into .ipynb and generates example gallery
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',   # nD: support for google style docstrings
    'sphinx.ext.githubpages', # JM add .nojekyll creation for github
    'sphinx.ext.doctest', # JM: adds doctest directives
    'sphinx_automodapi.automodapi', # makes separate doc pages for every object
    # 'sphinx_automodapi.smart_resolver', # don't think this is needed
    'sphinx_copybutton', # adds a copy button to our examples (NOT THE SAME AS copybutton.js)
    # 'nbsphinx', # JM: lets us include jupyter notebooks in sphinx rst files
    ]

# doctest config
doctest_global_setup = '''
import imagepypelines as ip
import doctest
doctest.ELLIPSIS_MARKER = "..."

IP_DEVNULL = None

def IP_SILENCE_STDOUT_STDERR():
    import sys, os
    # silence all standard output for global setup
    global IP_DEVNULL
    IP_DEVNULL = open(os.devnull, 'w')
    sys.stdout = IP_DEVNULL
    sys.stderr = IP_DEVNULL

def IP_RESET_STDOUT_STDERR():
    import sys
    global IP_DEVNULL
    if IP_DEVNULL is not None:
        IP_DEVNULL.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        IP_DEVNULL = None

# cleanup environment
del ip
'''
# JM - adds our custom landing page
html_additional_pages = {'index': 'index.html',
                          'about' : 'about.html',
                          # 'examples' : 'examples.html'
                          }
master_doc = 'index'

# JM - show inherited class attributes in automodapi
automodsumm_inherited_members = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_static/_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst',]


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None


# JM
def silence_ip(gallery_conf, fname):
    # increasing logging level so we are less likely
    import imagepypelines as ip
    ip.MASTER_LOGGER.setLevel("WARNING")
    # temporarily suppress Warnings (mostly for matplotlib)
    import warnings
    warnings.simplefilter('ignore')

sphinx_gallery_conf = {
    'examples_dirs': ['_examples'],
    'gallery_dirs': ['examples'],
    'filename_pattern': '/*',
    'ignore_pattern': r'__init__\.py',
    'reset_modules': (silence_ip,),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

# JM: nbsphinx (jupyter notebook support)
# from this guide: https://nbsphinx.readthedocs.io/en/0.6.0/usage.html
# exclude_patterns = ['_build', '**.ipynb_checkpoints']
# html_sourcelink_suffix = ''
#
# # JM: modified from here: https://github.com/spatialaudio/nbsphinx/blob/0.6.0/doc/conf.py#L45-L88
# nbsphinx_prolog = """
# Generated from `{{ env.docname }}.ipynb <https://github.com/jmaggio14/imagepypelines/tree/%s/docs/source/{{ env.docname }}>`_
#
# ----
# """ % GIT_SHA
# # This is processed by Jinja2 and inserted after each notebook
# nbsphinx_epilog = r"""
# {% set docname = 'doc/' + env.doc2path(env.docname, base=None) %}
# """


# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'abap'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'

# RH - Activate the pip imported theme
# JM - 04/02/20. commented out in favor of shipping with our own css
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# JM - adds documentation for __special__ functions with docstrings
napoleon_include_special_with_doc = True

# (Optional) Logo. Should be small enough to fit the navbar (ideally 32x32).
# Path should be relative to the ``_static`` files directory.
#
# RH -I'm currently working on this and will add at a later date - 09/29/18
#
html_logo = "./_static/images/ip_logo.svg"

html_favicon = "./_static/images/favicon.ico"

# Theme options are theme-specific and customize the look and feel of a
# theme further.
html_theme_options = {
    # Navigation bar title. (Default: ``project`` value)
    'navbar_title': "ImagePypelines",

    # Tab name for entire site. (Default: "Site")
    'navbar_site_name': "",

    # A list of tuples containing pages or urls to link to.
    # Valid tuples should be in the following forms:
    #    (name, page)                 # a link to a page
    #    (name, "/aa/bb", 1)          # a link to an arbitrary relative url
    #    (name, "http://example.com", True) # arbitrary absolute url
    # Note the "1" or "True" value above as the third argument to indicate
    # an arbitrary url.
    'navbar_links': [
                     ("About", 'about.html', True),
                     # ("Readme", 'readme.html', True),
                     ("Examples", 'examples/index.html', 1),
                     ("Documentation", 'docs/core'),
                     ("Plugins", 'plugins'),
                     ("Github", version_info['__download_url__'], True),
                     ],

    # Render the next and previous page links in navbar. (Default: true)
    'navbar_sidebarrel': False,

    # Render the current pages TOC in the navbar. (Default: true)
    'navbar_pagenav': False,

    # Tab name for the current pages TOC. (Default: "Page")
    'navbar_pagenav_name': "Page",

    # Global TOC depth for "site" navbar tab. (Default: 1)
    # Switching to -1 shows all levels.
    'globaltoc_depth': 0,

    # Include hidden TOCs in Site navbar?
    #
    # Note: If this is "false", you cannot have mixed ``:hidden:`` and
    # non-hidden ``toctree`` directives in the same page, or else the build
    # will break.
    #
    # Values: "true" (default) or "false"
    'globaltoc_includehidden': "true",

    # HTML navbar class (Default: "navbar") to attach to <div> element.
    # For black navbar, do "navbar navbar-inverse"
    'navbar_class': "navbar",

    # Fix navigation bar to top of page?
    # Values: "true" (default) or "false"
    'navbar_fixed_top': "false",

    # Location of link to source.
    # Options are "nav" (default), "footer" or anything else to exclude.
    'source_link_position': "footer",

    # Bootswatch (http://bootswatch.com/) theme.
    #
    # Options are nothing (default) or the name of a valid theme
    # such as "cosmo" or "sandstone".
    #
    # The set of valid themes depend on the version of Bootstrap
    # that's used (the next config option).
    #
    # Currently, the supported themes are:
    # - Bootstrap 2: https://bootswatch.com/2
    # - Bootstrap 3: https://bootswatch.com/3
    'bootswatch_theme': "united",

    # Choose Bootstrap version.
    # Values: "3" (default) or "2" (in quotes)
    'bootstrap_version': "3",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'imagepypelinesdoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'imagepypelines.tex', 'ImagePypelines Documentation',
     version_info['__author__'], 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'ImagePypelines', 'ImagePypelines Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'ImagePypelines', 'ImagePypelines Documentation',
     author, 'ImagePypelines', version_info['__description__'],
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------
