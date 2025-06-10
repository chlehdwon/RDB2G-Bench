Bayesian Optimization
=====================

This module implements Bayesian optimization baseline for neural architecture search.
Bayesian optimization is a sequential model-based optimization technique particularly
effective for expensive black-box optimization problems like finding optimal GNN architectures.

How it Works
------------

The Bayesian optimization process:

1. Build a probabilistic model (surrogate) of the objective function
2. Use acquisition function to determine next graph construction to evaluate
3. Evaluate the objective function at the selected graph construction
4. Update the surrogate model with new observation
5. Repeat until budget is exhausted

Bayesian optimization baseline
-----------

.. automodule:: rdb2g_bench.benchmark.baselines.bo
    :members:
    :undoc-members:
    :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from rdb2g_bench.benchmark.runner import run_benchmark

   # Run Bayesian optimization with default parameters
   results = run_benchmark(
       dataset="rel-f1",
       task="driver-top3", 
       budget_percentage=0.05,
       method=["bo"],
       num_runs=10,
       seed=42
   )
   
   # Access results
   print(f"Best architecture found: {results['bo']['selected_graph_id']}")
   print(f"Performance: {results['bo']['actual_y_perf_of_selected']:.4f}")
   print(f"Rank: {results['bo']['rank_position_overall']}")
