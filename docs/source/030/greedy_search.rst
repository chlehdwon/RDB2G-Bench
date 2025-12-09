Greedy Search
=============

This module implements multiple greedy search strategies for RDB-to-Graph modeling search.
Greedy algorithms make locally optimal choices at each step, providing fast and deterministic
approaches for finding good graph models with minimal computational overhead.

How it Works
------------

The greedy search algorithm implements three different greedy strategies for graph model optimization:

1. **Forward Greedy**: Starts with a graph model only with target table(s) and iteratively moves to the best graph model
2. **Backward Greedy**: Starts with the All-Rows-to-Nodes(AR2N) graph model and iteratively moves to the best graph model
3. **Local Greedy**: Combines forward and backward greedy strategies with randomly initialized graph model

The greedy search algorithm process:

1. Starts with a random graph model
2. Evaluates the performance of the graph model
3. Selects the best local improvement based on the chosen greedy strategy
4. Repeats until the budget is exhausted


Greedy Search Baseline
----------------------

.. automodule:: rdb2g_bench.benchmark.baselines.greedy
    :members:
    :undoc-members:
    :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from rdb2g_bench.benchmark.runner import run_benchmark

   # Run greedy search by default
   results = run_benchmark(
       dataset="rel-f1",
       task="driver-top3",
       gnn="GraphSAGE",
       budget_percentage=0.05,
       method=["greedy"],
       num_runs=10,
       seed=42
   )
   
   # Access results
   print(f"Best architecture found: {results['greedy']['selected_graph_id']}")
   print(f"Performance: {results['greedy']['actual_y_perf_of_selected']:.4f}")
   print(f"Greedy steps completed: {results['greedy']['total_iterations_run']}")
