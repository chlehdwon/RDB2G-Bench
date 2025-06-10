# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'RDB2G-Bench'
copyright = 'KAIST Data Mining Lab'
author = 'Dongwon Choi'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.todo'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

autodoc_mock_imports = ["torch", "dgl", "numpy", "os", "time", "copy", "itertools", "pickle", "torch_scatter", "sklearn", "ogb", "scipy", "networkx", "tqdm", "qpth", "quadprog", "cvxpy", "rdkit", "dgllife", "pandas"]


# -- Options for EPUB output
epub_show_urls = 'footnote'

add_module_names = False