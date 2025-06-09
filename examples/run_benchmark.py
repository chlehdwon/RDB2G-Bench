import os
os.environ["ANTHROPIC_API_KEY"] = "YOUR_API_KEY"

from rdb2g_bench.benchmark.bench_runner import run_benchmark
from rdb2g_bench.benchmark.llm.llm_runner import run_llm_baseline

# Example 1: Basic run
run_benchmark(
    dataset="rel-f1",
    task="driver-top3", 
    budget_percentage=0.05,
    method="all",
    num_runs=1,
    seed=0,
)

# Example 2: Run only for specific methods (Currently, Greedy, BO, RL, and EA are supported)
run_benchmark(
    dataset="rel-f1",
    task="driver-top3", 
    budget_percentage=0.05,
    method=["rl", "bo"],
    num_runs=1,
    seed=0,
)

# Example 3: Run LLM-based baseline
run_llm_baseline(
    dataset="rel-f1",
    task="driver-top3", 
    budget_percentage=0.05,
    seed=0,
)