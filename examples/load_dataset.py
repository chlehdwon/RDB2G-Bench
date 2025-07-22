from rdb2g_bench.dataset.dataset import load_rdb2g_bench

bench = load_rdb2g_bench(result_dir="./results")

available = bench.get_available()
print(available)
"""
{
    'rel-stack': {
        'post-post-related': ['GraphSAGE', 'GIN', 'GPS']
    },
    'rel-event': {
        'user-attendance': ['GraphSAGE', 'GIN', 'GPS'],
        'user-repeat': ['GraphSAGE', 'GIN', 'GPS'],
        'user-ignore': ['GraphSAGE', 'GIN', 'GPS']
    },
    'rel-f1': {
        'driver-top3': ['GraphSAGE', 'GIN', 'GPS'],
        'driver-position': ['GraphSAGE', 'GIN', 'GPS'],
        'driver-dnf': ['GraphSAGE', 'GIN', 'GPS']
    },
    'rel-trial': {
        'study-outcome': ['GraphSAGE', 'GIN', 'GPS']
    },
    'rel-avito': {
        'user-visits': ['GraphSAGE', 'GIN', 'GPS'],
        'ad-ctr': ['GraphSAGE', 'GIN', 'GPS'],
        'user-clicks': ['GraphSAGE', 'GIN', 'GPS'],
        'user-ad-visit': ['GraphSAGE', 'GIN', 'GPS']
    }
}
"""

# Access specific task and GNN model
task = bench['rel-f1']['driver-top3']
available_gnns = task.get_available_gnns()
print(f"Available GNNs: {available_gnns}") # ['GraphSAGE', 'GIN', 'GPS']

# Access specific GNN model
gnn = task['GraphSAGE']
indices = gnn.get_available_indices()
print(f"Available indices for GraphSAGE: {len(indices)}") # 722

# Get results for specific graph configuration
result = gnn[0]
print(f"Index 0 results: {result.stats}")
"""
Index 0: {'test_metric_mean': 0.805233155657748,
        'test_metric_std': 0.034900872955993194,
        'params': 954241,
        'train_time': 0.30655655304590856,
        'valid_time': 0.06576369603474932,
        'test_time': 0.05956562360127762}
"""

# Access aggregated statistics
stats = result.stats
test_metric_mean = stats['test_metric_mean']
test_metric_std = stats['test_metric_std']
params = stats['params']
train_time = stats['train_time']

print(f"Test metric: {test_metric_mean:.4f} ± {test_metric_std:.4f}") # 0.8052 ± 0.0347
print(f"Parameters: {params:.0f}") # 954241
print(f"Train time: {train_time:.4f} seconds") # 0.3066

# Compare different GNN models
print("\nComparing GNN models:")
for gnn_name in available_gnns:
    gnn_model = task[gnn_name]
    result_0 = gnn_model[0]
    perf = result_0.stats['test_metric_mean']
    print(f"{gnn_name}: {perf:.4f}")