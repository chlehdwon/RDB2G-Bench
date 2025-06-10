# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

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

autodoc_mock_imports = [
    "torch", "torch_frame", "torch_geometric", "torch_scatter", "torch_sparse", 
    "torch_cluster", "torch_spline_conv", "pyg_lib",
    "dgl", "numpy", "pandas", "sklearn", "scipy", "networkx", 
    "ogb", "tqdm", "qpth", "quadprog", "cvxpy", "rdkit", "dgllife",
    "relbench", "anthropic", "openai", 
    "typin", "pathlib", "json", "ast", "copy", "itertools", "pickle",
    "matplotlib", "seaborn", "plotly", "wandb", "tensorboard",
    "transformers", "datasets", "tokenizers", "accelerate",
    "optuna", "ray", "hyperopt", "ax-platform",
    "psutil", "memory_profiler", "line_profiler"
]


# -- Options for EPUB output
epub_show_urls = 'footnote'

add_module_names = False