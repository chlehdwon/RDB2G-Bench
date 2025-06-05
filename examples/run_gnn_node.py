from rdb2g_bench.dataset.node_worker import run_gnn_node_worker

# Example 1: Basic run
results = run_gnn_node_worker(
    dataset_name="rel-f1",
    task_name="driver-top3",
    gnn="GraphSAGE",
)

print(results['processed_graphs']) # [0, ..., 721]
print(results['total_processed']) # 722
print(results['csv_file']) # ./results/rel-f1/driver-top3/42.csv

# Example 2: Run parallelly on multiple GPUs
results_even = run_gnn_node_worker(
    dataset_name="rel-f1",
    task_name="driver-top3",
    gnn="GraphSAGE",
    idx=0,
    workers=2,
    device="cuda:0"
)

results_odd = run_gnn_node_worker(
    dataset_name="rel-f1",
    task_name="driver-top3",
    gnn="GraphSAGE",
    idx=1,
    workers=2,
    device="cuda:1"
)

# Example 3: Run worker on specific target indices
results_0 = run_gnn_node_worker(
    dataset_name="rel-f1",
    task_name="driver-top3",
    gnn="GraphSAGE",
    target_indices=[0]
)