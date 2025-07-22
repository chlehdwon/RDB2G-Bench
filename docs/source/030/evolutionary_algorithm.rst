Evolutionary Algorithm
======================

This module implements evolutionary algorithm baseline for neural architecture search.
Evolutionary algorithms are population-based metaheuristics that use biological evolution
mechanisms such as reproduction, mutation, and selection to find optimal solutions.

How it Works
------------

The evolutionary algorithm process:

1. Initialize a population of random graph constructions
2. Evaluate the performance of each graph construction
3. Select parents for reproduction
4. Apply crossover and mutation operators to the selected graph constructions
5. Replace old population with new generation
6. Repeat until budget is exhausted

Evolutionary Algorithm Baseline
-------------------------------

.. automodule:: rdb2g_bench.benchmark.baselines.ea
    :members:
    :undoc-members:
    :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from rdb2g_bench.benchmark.runner import run_benchmark

   # Run evolutionary algorithm with default parameters
   results = run_benchmark(
       dataset="rel-f1",
       task="driver-top3",
       gnn="GraphSAGE",
       budget_percentage=0.05,
       method=["ea"],
       num_runs=10,
       seed=42
   )
   
   # Access results
   print(f"Best architecture found: {results['ea']['selected_graph_id']}")
   print(f"Performance: {results['ea']['actual_y_perf_of_selected']:.4f}")
   print(f"Generations completed: {results['ea']['total_iterations_run']}")
