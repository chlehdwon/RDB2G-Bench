Random Search
=============

This module implements random search baseline for RDB-to-Graph modeling search.
Random search provides a simple yet effective baseline by uniformly sampling graph models
from the search space without any heuristic guidance, serving as an important comparison
point for evaluating more sophisticated optimization algorithms.

How it Works
------------

The random search algorithm process:

1. Uniformly sample graph models from the entire search space
2. Evaluate the performance of each sampled graph model
3. Keep track of the best graph model found so far
4. Repeat until the budget is exhausted

Random Search Baseline
----------------------

.. automodule:: rdb2g_bench.benchmark.baselines.random
    :members:
    :undoc-members:
    :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from rdb2g_bench.benchmark.runner import run_benchmark

   # Run random search baseline
   results = run_benchmark(
       dataset="rel-f1",
       task="driver-top3",
       gnn="GraphSAGE",
       budget_percentage=0.05,
       method=["random"],
       num_runs=10,
       seed=42
   )
   
   # Access results
   print(f"Best architecture found: {results['random']['selected_graph_id']}")
   print(f"Performance: {results['random']['actual_y_perf_of_selected']:.4f}")
   print(f"Architectures evaluated: {results['random']['discovered_count']}") 