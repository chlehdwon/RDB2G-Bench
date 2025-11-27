# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

# ReadTheDocs-specific configuration
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    # ReadTheDocs automatically installs the package via .readthedocs.yaml
    # But we need to ensure the path is correct
    sys.path.insert(0, os.path.abspath('../../'))

# -- Project information

project = 'RDB2G-Bench'
copyright = 'KAIST Data Mining Lab'
author = 'Dongwon Choi'

release = '0.1'
version = '0.1.1'

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
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# Show project logo at the top-left of the docs
html_logo = '../logo.png'

# HTML theme options for sphinx_rtd_theme
html_theme_options = {
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # These options are for the theme but we'll add GitHub link via html_context
}

# Add GitHub link in the HTML context
html_context = {
    'display_github': True,
    'github_user': 'chlehdwon',
    'github_repo': 'RDB2G-Bench',
    'github_version': 'main',
    'conf_py_path': '/docs/source/',
}

# -- Autodoc configuration for type hints
autodoc_typehints = 'signature'

autodoc_mock_imports = [
    "torch", "numpy", "pandas", "typing_extensions",
    "torch_frame", "torch_geometric", "torch_scatter", "torch_sparse", 
    "torch_cluster", "torch_spline_conv", "pyg_lib",
    "dgl", "sklearn", "scipy", "networkx", 
    "ogb", "tqdm", "qpth", "quadprog", "cvxpy", "rdkit", "dgllife",
    "relbench", "anthropic", "openai", 
    "typin", "pathlib", "json", "ast", "copy", "itertools", "pickle",
    "matplotlib", "seaborn", "plotly", "wandb", "tensorboard",
    "transformers", "datasets", "tokenizers", "accelerate",
    "optuna", "ray", "hyperopt", "ax-platform",
    "psutil", "memory_profiler", "line_profiler",
    # ReadTheDocs specific mocks
    "torch.nn", "torch.optim", "torch.utils", "torch.cuda"
]

if on_rtd:
    autodoc_mock_imports.extend([
        "torch.nn.functional", "torch.distributions",
        "torch_geometric.nn", "torch_geometric.data", "torch_geometric.utils",
        "relbench.base", "relbench.datasets", "relbench.modeling", "relbench.tasks"
    ])

# -- Options for EPUB output
epub_show_urls = 'footnote'

add_module_names = False