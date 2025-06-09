Reinforcement Learning
======================

Overview
--------

Reinforcement learning approaches treat hyperparameter optimization as a sequential decision-making problem.

Basic Usage
-----------

Run reinforcement learning benchmark:

.. code-block:: python

   from rdb2g_bench.benchmark.runner import run_benchmark

   # Run reinforcement learning approach
   results = run_benchmark(
       dataset="rel-f1",
       task="driver-top3", 
       budget_percentage=0.05,
       method=["rl"],
       num_runs=10,
       seed=42
   )

How it Works
------------

The RL-based optimization process:

1. Define the search space as an environment
2. Train an RL agent to navigate the space
3. Agent learns to select promising configurations
4. Reward is based on model performance
5. Agent improves its strategy over time

Key Components
--------------

**Environment**: The hyperparameter search space
**Agent**: The learning algorithm that selects configurations
**State**: Current configuration or search history
**Action**: Moving to a new configuration
**Reward**: Performance improvement or absolute performance

Common RL Algorithms
--------------------

- **Q-Learning**: Value-based method for discrete spaces
- **Policy Gradient**: Direct policy optimization
- **Actor-Critic**: Combines value and policy methods
- **Deep Q-Networks (DQN)**: Deep learning extension of Q-learning

Parameters
----------

Key parameters for RL-based optimization:

- ``algorithm``: Type of RL algorithm
- ``learning_rate``: Step size for policy updates
- ``exploration_rate``: Balance between exploration and exploitation
- ``network_architecture``: Neural network design for deep RL
