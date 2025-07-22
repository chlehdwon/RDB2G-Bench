from rdb2g_bench.dataset.dataset import download_rdb2g_bench, get_dataset_stats

# List all available datasets and tasks with GNN models
df_stats = get_dataset_stats(cache_dir="~/.cache")
print(df_stats)
"""
      dataset               task        gnn    idx  seed  test_metric_mean  test_metric_std  test_metric_min  test_metric_max                                                                                                                                               
0   rel-avito             ad-ctr  GraphSAGE   1304     5            0.0423           0.0010           0.0374           0.0458                                                                                                                                               
1   rel-avito             ad-ctr        GIN   1304     5            0.0419           0.0012           0.0365           0.0461                                                                                                                                               
2   rel-avito             ad-ctr        GPS   1304     5            0.0425           0.0009           0.0378           0.0459                                                                                                                                               
3   rel-avito      user-ad-visit  GraphSAGE    909     5            0.0165           0.0076           0.0016           0.0372                                                                                                                                               
4   rel-avito      user-ad-visit        GIN    909     5            0.0163           0.0074           0.0018           0.0369                                                                                                                                               
5   rel-avito      user-ad-visit        GPS    909     5            0.0168           0.0078           0.0014           0.0375                                                                                                                                               
6      rel-f1        driver-top3  GraphSAGE    722    15            0.7847           0.0150           0.6554           0.8732                                                                                                                                               
7      rel-f1        driver-top3        GIN    722    15            0.7823           0.0148           0.6521           0.8698                                                                                                                                               
8      rel-f1        driver-top3        GPS    722    15            0.7891           0.0152           0.6598           0.8756
...
"""

# Filter statistics by specific GNN model
graphsage_stats = df_stats[df_stats['gnn'] == 'GraphSAGE']
print("\nGraphSAGE performance across datasets:")
print(graphsage_stats[['dataset', 'task', 'test_metric_mean', 'test_metric_std']])

# Download entire RDB2G-Bench dataset (includes all GNN models)
saved_files = download_rdb2g_bench(
    result_dir="./results",
    cache_dir="~/.cache",
    tag="hf"
)
print(f"\nDownloaded {len(saved_files)} dataset/task combinations")
for combo, files in saved_files.items():
    print(f"{combo}: {len(files)} files")

# Download specific RDB2G-Bench dataset with specific GNN model
saved_files = download_rdb2g_bench(
    result_dir="./results",
    cache_dir="~/.cache",
    dataset_names=["rel-f1"],
    task_names=["driver-top3"],
    gnn_names=["GIN"],
    tag="hf_gin"
)
print(f"\nSpecific GNN model download saved files:")
for combo, files in saved_files.items():
    print(f"{combo}: {files}")