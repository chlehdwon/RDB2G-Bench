from rdb2g_bench.dataset.link_worker import run_idgnn_link_worker

# Example 1: Basic run
results = run_idgnn_link_worker(
    dataset_name="rel-avito",
    task_name="user-ad-visit",
    gnn="GraphSAGE",
)

print(results['processed_graphs']) # [0, ..., 908]
print(results['total_processed']) # 909
print(results['csv_file']) # ./results/rel-avito/user-ad-visit/42.csv

# Example 2: Run parallelly on multiple GPUs
results_even = run_idgnn_link_worker(
    dataset_name="rel-avito",
    task_name="user-ad-visit",
    gnn="GraphSAGE",
    idx=0,
    workers=2,
    device="cuda:0"
)

results_odd = run_idgnn_link_worker(
    dataset_name="rel-avito",
    task_name="user-ad-visit",
    gnn="GraphSAGE",
    idx=1,
    workers=2,
    device="cuda:1"
)

# Example 3: Run worker on specific target indices
results_0 = run_idgnn_link_worker(
    dataset_name="rel-avito",
    task_name="user-ad-visit",
    gnn="GraphSAGE",
    target_indices=[0]
)