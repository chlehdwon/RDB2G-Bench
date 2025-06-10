Tutorial
========

Basic Usage
-----------

This tutorial provides a quick overview of RDB2G-Bench.

Quick Start Example
-------------------

Here's a simple example to get you started:

.. code-block:: python

   from rdb2g_bench.dataset.dataset import load_rdb2g_bench

   # Load pre-computed benchmark results
   bench = load_rdb2g_bench("./results")
   
   # Access specific results
   result = bench['rel-f1']['driver-top3'][0]
   test_metric = result['test_metric']
   params = result['params']
   train_time = result['train_time']
   
   print(f"Test Metric: {test_metric}")
   print(f"Training Time: {train_time}")

Run First Benchmark
-----------------------------

.. code-block:: python

   from rdb2g_bench.benchmark.runner import run_benchmark

   # Run a simple benchmark
   results = run_benchmark(
       dataset="rel-f1",
       task="driver-top3", 
       budget_percentage=0.05,
       method="all",
       num_runs=3,
       seed=42
   )

For more detailed examples, check the ``examples/`` directory in the repository.

Package Structure Overview
--------------------------

.. code-block:: text

   rdb2g_bench/
   ├── benchmark/         # Core benchmarking functionality
   ├── common/            # Shared utilities and search spaces  
   ├── dataset/           # Dataset loading and processing
   └── __init__.py        # Package initialization 