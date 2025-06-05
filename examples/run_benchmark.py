from rdb2g_bench.benchmark.bench_runner import run_benchmark

# Example 1: Basic run
results = run_benchmark(
    dataset="rel-f1",
    task="driver-top3", 
    budget_percentage=0.05,
    method="all",
    num_runs=1,
    seed=0,
)

print(results)

# Example 2: Run only for specific methods
results_bo = run_benchmark(
    dataset="rel-f1",
    task="driver-top3", 
    budget_percentage=0.05,
    method=["rl", "bo"],
    num_runs=1,
    seed=0,
)

print(results_bo)