Bayesian Optimization
=====================

Overview
--------

Bayesian optimization is a sequential design strategy for global optimization of expensive-to-evaluate functions.

Basic Usage
-----------

Run Bayesian optimization benchmark:

.. code-block:: python

   from rdb2g_bench.benchmark.runner import run_benchmark

   # Run Bayesian optimization
   results = run_benchmark(
       dataset="rel-f1",
       task="driver-top3", 
       budget_percentage=0.05,
       method=["bo"],
       num_runs=10,
       seed=42
   )

How it Works
------------

The Bayesian optimization process:

1. Build a probabilistic model (surrogate) of the objective function
2. Use acquisition function to determine next point to evaluate
3. Evaluate the objective function at the selected point
4. Update the surrogate model with new observation
5. Repeat until budget is exhausted

Key Components
--------------

**Surrogate Model**: Probabilistic model of the objective function (e.g., Gaussian Process)
**Acquisition Function**: Strategy for selecting next evaluation point
**Optimization**: Finding the maximum of the acquisition function

Common Acquisition Functions
----------------------------

- **Expected Improvement (EI)**: Balance between exploration and exploitation
- **Upper Confidence Bound (UCB)**: Optimistic strategy
- **Probability of Improvement (PI)**: Conservative approach

Parameters
----------

Key parameters for Bayesian optimization:

- ``surrogate_model``: Type of surrogate model (GP, Random Forest, etc.)
- ``acquisition_function``: Strategy for point selection
- ``kernel``: Kernel function for Gaussian Process
- ``exploration_weight``: Balance between exploration and exploitation
