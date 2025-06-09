Download Pre-computed Datasets
==============================

Overview
--------

RDB2G-Bench provides pre-computed benchmark results that you can directly download and use without running experiments yourself.

Basic Usage
-----------

Load pre-computed benchmark results:

.. code-block:: python

   from rdb2g_bench.dataset.dataset import load_rdb2g_bench

   # Load benchmark results
   bench = load_rdb2g_bench("./results")

Accessing Results
-----------------

Access specific benchmark results by dataset and task:

.. code-block:: python

   # Access results for rel-f1 dataset, driver-top3 task
   result = bench['rel-f1']['driver-top3'][0]  # First graph configuration
   
   # Extract performance metrics
   test_metric = result['test_metric']         # Test performance
   params = result['params']                   # Model parameters
   train_time = result['train_time']           # Training time

