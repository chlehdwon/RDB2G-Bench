Run Benchmark
=============

This module provides the main interface for running comprehensive benchmarks on RDB2G-Bench.
The benchmark runner executes multiple neural architecture search algorithms and provides
statistical analysis and comparison across different methods, datasets, and tasks.

The benchmark system supports all available search strategies and automatically handles
data preparation, caching, multi-run execution, and results aggregation.

Benchmark Runner Interface
--------------------------

.. automodule:: rdb2g_bench.benchmark.bench_runner
    :members:
    :undoc-members:
    :show-inheritance:

Core Benchmark Engine
---------------------

.. automodule:: rdb2g_bench.benchmark.benchmark
    :members:
    :undoc-members:
    :show-inheritance:

Example Usage
~~~~~~~~~~~~~

Basic Benchmark Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from rdb2g_bench.benchmark.bench_runner import run_benchmark

   # Run all methods on default dataset and task
   results = run_benchmark(
       dataset="rel-f1",
       task="driver-top3",
       budget_percentage=0.05,
       method="all",
       num_runs=10,
       seed=42
   )
   
   # Print summary of results
   for method, stats in results.items():
       if 'avg_actual_y_perf_of_selected' in stats:
           perf = stats['avg_actual_y_perf_of_selected']
           rank = stats.get('avg_rank_position_overall', 'N/A')
           print(f"{method}: Performance={perf:.4f}, Rank={rank}")

Specific Methods Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Compare evolutionary algorithm vs greedy search
   results = run_benchmark(
       dataset="rel-avito",
       task="user-ad-visit", 
       budget_percentage=0.05,
       method=["ea", "greedy", "random"],
       num_runs=15,
       seed=123
   )
   
   # Extract key metrics
   for method in ["ea", "greedy", "random"]:
       if method in results:
           stats = results[method]
           print(f"\n{method.upper()} Results:")
           print(f"  Average Performance: {stats['avg_actual_y_perf_of_selected']:.4f}")
           print(f"  Average Rank: {stats['avg_rank_position_overall']:.1f}")
           print(f"  Average Runtime: {stats['avg_run_time']:.2f}s")

Available Methods
-----------------

The benchmark runner supports the following search methods:

- **"all"**: Run all available methods
- **"ea"**: Evolutionary Algorithm baseline
- **"greedy"**: Greedy Search strategies (forward, backward, random)
- **"rl"**: Reinforcement Learning with policy gradients
- **"bo"**: Bayesian Optimization with surrogate models

You can specify a single method as a string or multiple methods as a list.

Results Structure
-----------------

The returned results dictionary contains method-wise statistics:

.. code-block:: python

   {
       "Method Name": {
           "avg_actual_y_perf_of_selected": float,      # Average performance
           "avg_rank_position_overall": float,          # Average ranking
           "avg_percentile_overall": float,             # Average percentile
           "total_samples_overall": int,                # Total architectures
           "selected_graph_ids_runs": list,             # Selected graph IDs
           "avg_selection_metric_value": float,         # Average selection metric
           "selected_graph_origins": list,              # Method origins
           "avg_evaluation_time": float,                # Average evaluation time
           "avg_run_time": float                        # Average total runtime
       }
   }

Output Files
------------

The benchmark automatically generates several output files:

- **Individual Trajectories**: ``avg_trajectory_{method}_{num_runs}runs.csv``
- **Combined Trajectories**: ``all_methods_trajectories_{num_runs}runs.csv``  
- **Performance Summary**: ``performance_summary_{num_runs}runs.csv``

These files are saved in: ``{result_dir}/benchmark/{dataset}/{task}/{tag}/``
