from rdb2g_bench.dataset.dataset import download_rdb2g_bench, get_dataset_stats

# List all available datasets and tasks
df_stats = get_dataset_stats(cache_dir="~/.cache")
print(df_stats)
"""
      dataset               task    idx  seed  test_metric_mean  test_metric_std  test_metric_min  test_metric_max                                                                                                                                               
0   rel-avito             ad-ctr   1304     5            0.0423           0.0010           0.0374           0.0458                                                                                                                                               
1   rel-avito      user-ad-visit    909     5            0.0165           0.0076           0.0016           0.0372                                                                                                                                               
2   rel-avito        user-clicks    944     5            0.6496           0.0123           0.5906           0.6873                                                                                                                                               
3   rel-avito        user-visits    944     5            0.6331           0.0113           0.5845           0.6656                                                                                                                                               
4   rel-event    user-attendance    214    15            0.2491           0.0108           0.2311           0.2699                                                                                                                                               
5   rel-event        user-ignore    214    15            0.7864           0.0248           0.6990           0.8707                                                                                                                                               
6   rel-event        user-repeat    214    15            0.7540           0.0636           0.5490           0.8419                                                                                                                                               
7      rel-f1         driver-dnf    722    15            0.7176           0.0196           0.5999           0.7763                                                                                                                                               
8      rel-f1    driver-position    722    15            3.8791           0.0568           3.7578           4.4220                                                                                                                                               
9      rel-f1        driver-top3    722    15            0.7847           0.0150           0.6554           0.8732                                                                                                                                               
10  rel-stack  post-post-related   7979     5            0.0461           0.0518           0.0001           0.1241                                                                                                                                               
11  rel-trial      study-outcome  36863     5            0.6745           0.0148           0.5803           0.7214  
"""

# Download entire RDB2G-Bench dataset
saved_files = download_rdb2g_bench(
    result_dir="./results",
    cache_dir="~/.cache",
    tag="hf"
)
print(saved_files)

# Download specific RDB2G-Bench dataset
saved_files = download_rdb2g_bench(
    result_dir="./results",
    cache_dir="~/.cache",
    dataset_names=["rel-f1"],
    task_names=["driver-top3"],
    tag="hf_top3"
)
print(saved_files)