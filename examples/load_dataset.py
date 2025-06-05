from rdb2g_bench.dataset.dataset import load_rdb2g_bench

bench = load_rdb2g_bench(result_dir="./results")

available = bench.get_available()
print(available)
"""
{
    'rel-stack': ['post-post-related'],
    'rel-event': ['user-attendance', 'user-repeat', 'user-ignore'],
    'rel-f1': ['driver-top3', 'driver-position', 'driver-dnf'],
    'rel-trial': ['study-outcome'],
    'rel-avito': ['user-visits', 'ad-ctr', 'user-clicks', 'user-ad-visit']
}
"""

task = bench['rel-f1']['driver-top3']
indices = task.get_available_indices()
print(len(indices)) # 722

result = task[0]
print(f"Index 0 results: {result}")
"""
Index 0: {'test_metric_mean': 0.805233155657748,
        'test_metric_std': 0.034900872955993194,
        'params': 954241,
        'train_time': 0.30655655304590856,
        'valid_time': 0.06576369603474932,
        'test_time': 0.05956562360127762}
"""

test_metric = result['test_metric']
params = result['params']
train_time = result['train_time']

print(f"Test metric: {test_metric:.4f}") # 0.8052
print(f"Parameters: {params:.0f}") # 954241
print(f"Train time: {train_time:.4f} seconds") # 0.3066