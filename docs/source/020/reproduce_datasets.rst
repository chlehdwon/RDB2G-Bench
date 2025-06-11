Reproduce Datasets
==================

This module provides functionality to reproduce the datasets and benchmark results from the RDB2G-Bench paper.
The workers support both node-level tasks (classification/regression) and link prediction tasks (recommendation)
with various GNN architectures and comprehensive evaluation metrics.

Node Worker
-----------

The node worker handles node classification and regression tasks using Graph Neural Networks.

.. automodule:: rdb2g_bench.dataset.node_worker
    :members:
    :undoc-members:
    :show-inheritance:

Link Worker
-----------

The link worker handles link prediction tasks using ID-aware Graph Neural Networks (IDGNN).

.. automodule:: rdb2g_bench.dataset.link_worker
    :members:
    :undoc-members:
    :show-inheritance:

Task Types
----------

The benchmark supports different types of tasks:

Node-Level Tasks
~~~~~~~~~~~~~~~~

- **Binary Classification**: Predicting binary labels for nodes (e.g., driver performance)
- **Regression**: Predicting continuous values for nodes (e.g., ratings, scores)
- **Multilabel Classification**: Predicting multiple binary labels per node

Link Prediction Tasks
~~~~~~~~~~~~~~~~~~~~~

- **Recommendation**: Predicting user-item interactions using ranking metrics
- **Link Prediction**: General link prediction between entities

Supported Models
----------------

Both workers support multiple GNN architectures:

- **GraphSAGE**: Inductive graph representation learning
- **GIN**: Graph Isomorphism Network for graph-level tasks
- **GPS**: Graph transformer with positional encodings

Example Usage
----------------

Node Classification/Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rdb2g_bench.dataset.node_worker import run_gnn_node_worker

   # Run binary classification experiment
   results = run_gnn_node_worker(
       dataset_name="rel-f1",
       task_name="driver-top3",
       gnn_model="GraphSAGE",
       epochs=20,
       lr=0.005,
       batch_size=512,
       channels=128,
   )
   print(f"Processed {results['total_processed']} graph configurations")

   # Run regression experiment
   results = run_gnn_node_worker(
       dataset_name="rel-f1",
       task_name="driver-position",
       gnn_model="GIN",
       epochs=50,
       lr=0.001,
       weight_decay=1e-4
   )

   # Parallel processing with multiple workers
   results = run_gnn_node_worker(
       dataset_name="rel-f1",
       task_name="driver-top3",
       idx=0,        # Worker 0 ~ 3
       workers=4,    # Total 4 workers
       epochs=20
   )

Link Prediction
~~~~~~~~~~~~~~~

.. code-block:: python

   from rdb2g_bench.dataset.link_worker import run_idgnn_link_worker

   # Run recommendation experiment
   results = run_idgnn_link_worker(
       dataset_name="rel-avito",
       task_name="user-ad-visit",
       gnn_model="GraphSAGE",
       epochs=20,
       lr=0.001,
       batch_size=512,
       channels=128,
       temporal_strategy="last",
   )

   # Run with specific graph configurations
   results = run_idgnn_link_worker(
       dataset_name="rel-avito",
       task_name="user-ad-visit",
       target_indices=[0, 5, 10, 15],  # Run only these configurations
       epochs=30,
       patience=10
   )


Parallel Processing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import multiprocessing as mp
   from concurrent.futures import ProcessPoolExecutor

   def run_worker(worker_id, total_workers):
       """Run a single worker process."""
       return run_gnn_node_worker(
           dataset_name="rel-f1",
           task_name="driver-top3",
           idx=worker_id,
           workers=total_workers,
           epochs=20,
           save_csv=True
       )

   # Run multiple workers in parallel
   num_workers = 4
   with ProcessPoolExecutor(max_workers=num_workers) as executor:
       futures = [
           executor.submit(run_worker, i, num_workers) 
           for i in range(num_workers)
       ]
       
       # Collect results
       all_results = [future.result() for future in futures]
       total_processed = sum(r['total_processed'] for r in all_results)
       print(f"Total graphs processed: {total_processed}")
