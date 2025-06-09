LLM-based Baseline
==================

Overview
--------

Large Language Model (LLM) based baselines leverage the reasoning capabilities of large language models for DB2Graph modeling search.

Setup
-----

Before using LLM-based baselines, set up your API key:

.. code-block:: bash

   export ANTHROPIC_API_KEY="YOUR_API_KEY"

Basic Usage
-----------

Run LLM-based baseline:

.. code-block:: python

   from rdb2g_bench.benchmark.llm.llm_runner import run_llm_baseline

   # Run LLM-based optimization
   results = run_llm_baseline(
       dataset="rel-f1",
       task="driver-top3",
       budget_percentage=0.05,
       model="claude-3-5-sonnet-latest",
       temperature=0.8,
       seed=42
   )

How it Works
------------

The LLM-based optimization process:

1. Provide the LLM with problem description and search space
2. LLM suggests promising hyperparameter configurations
3. Evaluate suggested configurations on the actual task
4. Provide feedback to the LLM about performance
5. LLM refines its suggestions based on feedback
6. Repeat until budget is exhausted

Parameters
----------

Key parameters for LLM-based optimization:

- ``model``: Which LLM model to use
- ``temperature``: Sampling temperature for response generation
- ``max_tokens``: Maximum tokens for LLM responses
- ``prompt_strategy``: How to format prompts for the LLM

