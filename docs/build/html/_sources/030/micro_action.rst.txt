Micro Actions
=============

This module defines the core micro actions used by all optimization algorithms for graph construction.
These atomic operations enable systematic exploration of the graph construction space by transforming
one valid configuration into another.

How it Works
------------

Micro actions represent atomic operations for transforming graph constructions:

1. **add_fk_pk_edge**: Add foreign key-primary key edge between tables
2. **remove_fk_pk_edge**: Remove foreign key-primary key edge between tables  
3. **convert_row_to_edge**: Convert row representation to edge representation
4. **convert_edge_to_row**: Convert edge representation to row representation

Each action transforms the current graph construction (edge set) to a new valid graph construction, enabling systematic exploration of the graph construction space.

Micro Action Set
----------------

.. automodule:: rdb2g_bench.benchmark.micro_action
    :members:
    :undoc-members:
    :show-inheritance:

Example Usage
~~~~~~~~~~~~~

Example with Data Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import os
   import json
   from pathlib import Path
   from torch_frame import stype
   from torch_frame.config.text_embedder import TextEmbedderConfig
   from relbench.datasets import get_dataset
   from relbench.tasks import get_task
   from relbench.modeling.graph import make_pkey_fkey_graph
   from relbench.modeling.utils import get_stype_proposal

   from rdb2g_bench.benchmark.micro_action import MicroActionSet
   from rdb2g_bench.common.search_space.gnn_search_space import GNNNodeSearchSpace
   from rdb2g_bench.common.text_embedder import GloveTextEmbedding

   # Step 1: Load dataset and task
   dataset_name = "rel-f1"
   task_name = "driver-top3"
   
   dataset = get_dataset(dataset_name, download=True)
   task = get_task(dataset_name, task_name, download=True)
   
   # Step 2: Prepare column type information
   cache_dir = os.path.expanduser("~/.cache/relbench_examples")
   stypes_cache_path = Path(f"{cache_dir}/{dataset_name}/stypes.json")
   
   try:
       with open(stypes_cache_path, "r") as f:
           col_to_stype_dict = json.load(f)
       for table, col_to_stype in col_to_stype_dict.items():
           for col, stype_str in col_to_stype.items():
               col_to_stype[col] = stype(stype_str)
   except FileNotFoundError:
       col_to_stype_dict = get_stype_proposal(dataset.get_db())
       Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
       with open(stypes_cache_path, "w") as f:
           json.dump(col_to_stype_dict, f, indent=2, default=str)
   
   # Step 3: Create heterogeneous graph data
   device = "cuda" if torch.cuda.is_available() else "cpu"
   hetero_data, col_stats_dict = make_pkey_fkey_graph(
       dataset.get_db(),
       col_to_stype_dict=col_to_stype_dict,
       text_embedder_cfg=TextEmbedderConfig(
           text_embedder=GloveTextEmbedding(device=device), 
           batch_size=256
       ),
       cache_dir=f"{cache_dir}/{dataset_name}/materialized",
   )
   
   # Step 4: Initialize micro action set
   micro_actions = MicroActionSet(
       dataset=dataset_name,
       task=task_name,
       hetero_data=hetero_data,
       GNNSpaceClass=GNNNodeSearchSpace,
       num_layers=2,
       src_entity_table=task.entity_table
   )

   print(f"Total number of valid graph configurations: {len(micro_actions.valid_edge_sets_list)}")
   print(f"Number of FK-PK edges: {len(micro_actions.fk_pk_indices)}")
   print(f"Number of R2E edges: {len(micro_actions.r2e_indices)}")

Basic Micro Action Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get the first valid edge set as starting point
   current_edge_set = micro_actions.valid_edge_sets_list[0]
   print(f"Starting edge set: {current_edge_set}")
   
   # Explore all possible FK-PK edge additions
   add_fk_actions = micro_actions.add_fk_pk_edge(current_edge_set)
   print(f"Possible FK-PK additions: {len(add_fk_actions)}")
   
   for new_set, index in add_fk_actions[:3]:  # Show first 3
       print(f"  Add action: {current_edge_set} -> {new_set} (index: {index})")
   
   # Explore FK-PK edge removals
   remove_fk_actions = micro_actions.remove_fk_pk_edge(current_edge_set)
   print(f"Possible FK-PK removals: {len(remove_fk_actions)}")
   
   # Explore row-to-edge conversions
   row_to_edge_actions = micro_actions.convert_row_to_edge(current_edge_set)
   print(f"Possible row-to-edge conversions: {len(row_to_edge_actions)}")
   
   # Explore edge-to-row conversions
   edge_to_row_actions = micro_actions.convert_edge_to_row(current_edge_set)
   print(f"Possible edge-to-row conversions: {len(edge_to_row_actions)}")
