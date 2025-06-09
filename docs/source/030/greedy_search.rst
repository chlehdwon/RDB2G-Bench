Greedy Search
=============

Overview
--------

Greedy search is a simple optimization method that makes locally optimal choices at each step.

Basic Usage
-----------

Run greedy search benchmark:

.. code-block:: python

   from rdb2g_bench.benchmark.runner import run_benchmark

   # Run greedy search
   results = run_benchmark(
       dataset="rel-f1",
       task="driver-top3", 
       budget_percentage=0.05,
       method="greedy",
       num_runs=10,
       seed=42
   )

How it Works
------------

The greedy search algorithm:

1. Starts with a random configuration
2. Evaluates neighboring configurations
3. Selects the best improvement
4. Repeats until no improvement is found

Parameters
----------

Key parameters for greedy search:

- ``budget_percentage``: Computational budget as percentage of total search space
- ``num_runs``: Number of independent runs
- ``seed``: Random seed for reproducibility
