Evolutionary Algorithm
======================

Overview
--------

Evolutionary algorithms are population-based optimization methods inspired by biological evolution.

Basic Usage
-----------

Run evolutionary algorithm benchmark:

.. code-block:: python

   from rdb2g_bench.benchmark.runner import run_benchmark

   # Run evolutionary algorithm
   results = run_benchmark(
       dataset="rel-f1",
       task="driver-top3", 
       budget_percentage=0.05,
       method=["evolutionary"],
       num_runs=10,
       seed=42
   )

How it Works
------------

The evolutionary algorithm process:

1. Initialize a population of random configurations
2. Evaluate fitness of each individual
3. Select parents for reproduction
4. Apply crossover and mutation operators
5. Replace old population with new generation
6. Repeat until budget is exhausted

Key Concepts
------------

**Population**: Set of candidate solutions
**Fitness**: Performance metric for each candidate
**Selection**: Choosing parents for reproduction
**Crossover**: Combining two parents to create offspring
**Mutation**: Random changes to introduce diversity

Parameters
----------

Key parameters for evolutionary algorithms:

- ``population_size``: Number of individuals in population
- ``crossover_rate``: Probability of crossover operation
- ``mutation_rate``: Probability of mutation operation
- ``selection_method``: Strategy for parent selection
