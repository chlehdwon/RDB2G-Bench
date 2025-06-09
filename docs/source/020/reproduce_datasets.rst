Reproduce Our Datasets
======================

Overview
--------

This section explains how to reproduce the datasets and benchmark results from the RDB2G-Bench paper.

Classification & Regression Tasks
----------------------------------

Use the node worker to reproduce classification and regression experiments:

.. code-block:: python

   from rdb2g_bench.dataset.node_worker import run_gnn_node_worker

   # Run classification/regression experiment
   results = run_gnn_node_worker(
       dataset="rel-f1",
       task="driver-top3",
       gnn_model="GraphSAGE",
       epochs=20,
       lr=0.005
   )

Supported GNN Models:
- ``GraphSAGE``
- ``GIN``
- ``GPS``

Recommendation Tasks
--------------------

Use the link worker for recommendation experiments:

.. code-block:: python

   from rdb2g_bench.dataset.link_worker import run_idgnn_link_worker

   # Run recommendation experiment
   results = run_idgnn_link_worker(
       dataset="rel-avito",
       task="user-ad-visit",
       gnn_model="GraphSAGE",
       epochs=20,
       lr=0.001
   )

Configuration Options
---------------------

Key parameters you can adjust:

- ``dataset``: Choose from available datasets
- ``task``: Specific prediction task
- ``gnn_model``: GNN architecture to use
- ``epochs``: Number of training epochs
- ``lr``: Learning rate
