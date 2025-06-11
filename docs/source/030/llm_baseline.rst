LLM-based Baseline
==================

This module implements Large Language Model (LLM) based baseline.

How it Works
------------

The LLM-based optimization process:

1. Provide the LLM with problem description and search space
2. LLM suggests promising micro actions for graph constructions
3. Evaluate suggested micro actions on the actual task
4. Provide feedback to the LLM about performance
5. LLM refines its suggestions based on feedback
6. Repeat until the budget is exhausted

LLM-Based baseline
------------------

.. automodule:: rdb2g_bench.benchmark.llm.llm_runner
    :members:
    :undoc-members:
    :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from rdb2g_bench.benchmark.llm.llm_runner import run_llm_baseline

   # Run LLM-based baseline
   results = run_llm_baseline(
       dataset="rel-f1",
       task="driver-top3", 
       budget_percentage=0.05,
       seed=42,
       model="claude-3-5-sonnet-latest",
       temperature=0.8,
   )

