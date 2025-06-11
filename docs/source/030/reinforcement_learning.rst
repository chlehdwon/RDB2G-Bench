Reinforcement Learning
======================

This module implements reinforcement learning baseline for neural architecture search.
The approach uses deep reinforcement learning with policy gradients to train
a recurrent neural network controller that learns to generate sequences of micro actions
for constructing high-performing graph neural network architectures.

How it Works
------------

The reinforcement learning algorithm process:

1. Initialize RNN-based controller to learn graph construction actions
2. Start from random graph construction and convert to state embedding
3. Controller selects actions
4. Evaluate performance of new graph construction and compute reward
5. Update controller policy using Policy Gradient based on discounted rewards
6. Repeat episodes until budget is exhausted

Reinforcement Learning Baseline
-------------------------------

.. automodule:: rdb2g_bench.benchmark.baselines.rl
    :members:
    :undoc-members:
    :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from rdb2g_bench.benchmark.runner import run_benchmark

   # Run reinforcement learning with default parameters
   results = run_benchmark(
       dataset="rel-f1",
       task="driver-top3", 
       budget_percentage=0.05,
       method=["rl"],
       num_runs=10,
       seed=42
   )
   
   # Access results
   print(f"Best architecture found: {results['rl']['selected_graph_id']}")
   print(f"Performance: {results['rl']['actual_y_perf_of_selected']:.4f}")
   print(f"Episodes completed: {results['rl']['total_iterations_run']}")
