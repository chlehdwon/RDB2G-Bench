Install and Setup
==================

Installation
------------

Clone the repository and install the package:

.. code-block:: bash

   git clone https://github.com/chlehdwon/RDB2G-Bench.git
   cd RDB2G-Bench
   pip install -e .

PyTorch Geometric Dependencies
------------------------------

Install additional PyG dependencies. The example below shows installation for torch 2.1.0 + cuda 12.1:

.. code-block:: bash

   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

.. note::
   You can skip this step if you don't want to reproduce the datasets.

Environment Setup
-----------------

For LLM-based baselines, set up your API key:

.. code-block:: bash

   export ANTHROPIC_API_KEY="YOUR_API_KEY"

Verification
------------

Verify your installation by importing the package:

.. code-block:: python

   import rdb2g_bench
   print("RDB2G-Bench installed successfully!") 