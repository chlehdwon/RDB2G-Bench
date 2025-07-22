Download Pre-computed Datasets
===============================

This module provides functionality to load and access pre-computed benchmark results without running experiments.
The RDB2G-Bench dataset can be downloaded from Hugging Face Hub and accessed through a hierarchical interface.

The RDB2G-Bench dataset is also available from the HuggingFace: https://huggingface.co/datasets/kaistdata/RDB2G-Bench

Dataset Module
--------------

The dataset module provides functions for downloading and loading benchmark data.

.. automodule:: rdb2g_bench.dataset.dataset
    :members:
    :undoc-members:
    :show-inheritance:

Dataloader Module
-----------------

The dataloader module provides classes for hierarchical access to benchmark results.

.. automodule:: rdb2g_bench.dataset.dataloader
    :members:
    :undoc-members:
    :show-inheritance:

Data Access Pattern
-------------------

The benchmark data follows a hierarchical access pattern:

.. code-block:: text

   RDB2GBench[dataset_name][task_name][gnn_name][idx] -> IndexAccessor
   
   Example:
   bench['rel-f1']['driver-top3']['GraphSAGE'][0] -> Results for graph configuration 0 with GraphSAGE

Directory Structure
~~~~~~~~~~~~~~~~~~~

The downloaded data is organized in the following directory structure:

.. code-block:: text

   results/
     tables/
       dataset_name/               # e.g., rel-f1, rel-avito
         task_name/                # e.g., driver-top3, user-ad-visit
           tag/                    # e.g., hf
             gnn_name/             # e.g., GraphSAGE, GIN, GPS
               0.csv               # Results for seed 0
               1.csv               # Results for seed 1
               ...

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from rdb2g_bench.dataset.dataset import load_rdb2g_bench

   # Load benchmark results (downloads if missing)
   bench = load_rdb2g_bench("./results")

   # Check available datasets, tasks, and GNN models
   available = bench.get_available()
   print(available)
   # {'rel-f1': {'driver-top3': ['GIN', 'GPS', 'GraphSAGE'], ...}, ...}

   # Access results for rel-f1 dataset, driver-top3 task, GraphSAGE model
   result = bench['rel-f1']['driver-top3']['GraphSAGE'][0]  # First graph configuration
   
   # Extract performance metrics (aggregated across seeds)
   test_metric = result['test_metric']         # Test performance mean
   test_std = result['test_metric_std']        # Test performance std
   params = result['params']                   # Model parameters
   train_time = result['train_time']           # Training time

CSV File Format
~~~~~~~~~~~~~~~

Each CSV file contains the following columns:

.. code-block:: text

   idx          : Graph configuration index
   graph        : Graph structure representation (e.g., "graph_00000000000010")
   train_metric : Training performance metric
   valid_metric : Validation performance metric  
   test_metric  : Test performance metric
   params       : Number of trainable parameters
   train_time   : Average training time per epoch
   valid_time   : Validation time
   test_time    : Test time
   dataset      : Dataset name
   task         : Task name
   seed         : Random seed
   gnn          : GNN model name (GraphSAGE, GIN, GPS)

Accessing GNN-specific Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The benchmark now supports GNN-specific access patterns:

.. code-block:: python

   # Access specific GNN models
   graphsage_result = bench['rel-f1']['driver-top3']['GraphSAGE'][0]
   gin_result = bench['rel-f1']['driver-top3']['GIN'][0]
   
   # Get available GNN models for a task
   available_gnns = bench['rel-f1']['driver-top3'].get_available_gnns()
   print(available_gnns)  # ['GIN', 'GPS', 'GraphSAGE']
   
   # Compare performance across GNN models
   for gnn in available_gnns:
       result = bench['rel-f1']['driver-top3'][gnn][0]
       print(f"{gnn}: {result['test_metric']}")



