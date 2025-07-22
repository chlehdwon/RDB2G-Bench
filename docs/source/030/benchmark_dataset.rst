Performance Prediction Dataset
===============================

This module implements the ``PerformancePredictionDataset`` class, which is used to load performance data for benchmarking RDB-to-Graph search algorithms on RDB2G-Bench.

The ``PerformancePredictionDataset`` class is the core data interface for RDB2G-Bench. It loads pre-computed performance results from CSV files, processes graph configurations, and provides a unified interface for benchmark algorithms to access performance data.

How it Works
------------

The dataset loading process involves the following steps:

1. **RelBench Integration**: Connects to the specified RelBench dataset and task
2. **Graph Materialization**: Creates heterogeneous graph data with proper embeddings
3. **Results Loading**: Reads performance data from CSV files for the specified GNN model
4. **Data Aggregation**: Groups results by graph configuration and aggregates across seeds
5. **Graph Indexing**: Creates mappings between graph configurations and search space

Performance Prediction Dataset
-------------------------------

.. automodule:: rdb2g_bench.benchmark.dataset
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

   from rdb2g_bench.benchmark.dataset import PerformancePredictionDataset
   
   # Initialize dataset
   dataset = PerformancePredictionDataset(
       dataset_name="rel-f1",
       task_name="driver-top3",
       gnn="GraphSAGE",
       tag="hf",
       result_dir="./results"
   )
   
   # Access basic information
   print(f"Number of configurations: {len(dataset)}")
   print(f"Target metric: {dataset.target_col}")
   print(f"Task type: {dataset.task.task_type}")



