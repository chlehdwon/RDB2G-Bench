Greedy Search
=============

This module implements multiple greedy search strategies for neural architecture search.
Greedy algorithms make locally optimal choices at each step, providing fast and deterministic
approaches for finding good graph neural network architectures with minimal computational overhead.

How it Works
------------

The greedy search algorithm implements three different greedy strategies for graph construction optimization:

1. **Forward Greedy**: Starts with a random graph construction and iteratively moves to the best neighboring construction
2. **Backward Greedy**: Starts with the full graph construction and iteratively removes edges to find the best construction
3. **Local Greedy**: Combines forward and backward greedy strategies with random graph construction

The greedy search algorithm:
1. Starts with a random graph construction
2. Evaluates the performance of the graph construction
3. Selects the best improvement based on the chosen greedy strategy
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
       budget_percentage=0.05,
       method=["greedy"],
       num_runs=10,
       seed=42
   )
   
   # Access results
   print(f"Best architecture found: {results['greedy']['selected_graph_id']}")
   print(f"Performance: {results['greedy']['actual_y_perf_of_selected']:.4f}")
   print(f"Greedy steps completed: {results['greedy']['total_iterations_run']}")
