import os
from typing import Dict, List, Union, Optional, Any

from .benchmark import main as benchmark_main


def run_benchmark(
    dataset: str = "rel-f1",
    task: str = "driver-top3",
    budget_percentage: float = 0.05,
    method: Union[str, List[str]] = "all",
    num_runs: int = 10,
    seed: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute comprehensive benchmark analysis on specified dataset and task.
    
    This function provides a high-level interface for running neural architecture search
    benchmarks on RDB2G-Bench. It supports multiple optimization methods and automatically
    handles data preparation, model training, evaluation, and results aggregation across
    multiple runs for statistical robustness.
    
    The benchmark process includes:
    
    1. Dataset and task preparation with proper caching
    2. Search space initialization with micro actions
    3. Performance prediction dataset setup
    4. Multiple independent runs with different random seeds
    5. Statistical analysis and results aggregation
    6. Trajectory analysis and CSV export for visualization
    
    Args:
        dataset (str): Name of the RelBench dataset to benchmark on.
            Available datasets include "rel-f1", "rel-avito", "rel-amazon", etc.
            Defaults to "rel-f1".
        task (str): Name of the RelBench task to evaluate.
            Task names depend on the dataset (e.g., "driver-top3", "user-ad-visit").
            Defaults to "driver-top3".
        budget_percentage (float): Budget percentage for search algorithms as fraction
            of total search space (0.0-1.0). Higher values allow more thorough search
            but increase computational cost. Defaults to 0.05 (5%).
        method (Union[str, List[str]]): Search method(s) to benchmark.
            Options: "all", "ea", "greedy", "rl", "bo", "random", or list of methods.
            "all" runs all available methods. Defaults to "all".
        num_runs (int): Number of independent runs for statistical analysis.
            More runs provide better statistical confidence but increase runtime.
            Defaults to 10.
        seed (int): Base random seed for reproducibility. Each run uses seed + run_index.
            Defaults to 0.
        **kwargs: Additional configuration parameters:
        
            - tag (str): Experiment tag for result organization. Defaults to "hf".
            - cache_dir (str): Directory for caching datasets and models. 
              Defaults to "~/.cache/relbench_examples".
            - result_dir (str): Root directory for saving results and trajectories.
              Defaults to "./results".
        
    Returns:
        Dict[str, Any]: Dictionary containing comprehensive benchmark results.
        
        - avg_actual_y_perf_of_selected (float): Average performance across runs
        - avg_rank_position_overall (float): Average ranking position
        - avg_percentile_overall (float): Average percentile ranking
        - selected_graph_ids_runs (List[int]): Graph IDs selected in each run
        - avg_evaluation_time (float): Average time spent on evaluations
        - avg_run_time (float): Average total runtime per run
        - method_wise_statistics (Dict): Detailed statistics for each method
        - performance_trajectories (Dict): Performance over time for each method
        - statistical_comparisons (Dict): Rankings and comparisons between methods

    Example:
        >>> # Run all methods on default dataset/task
        >>> results = run_benchmark(
        ...     dataset="rel-f1",
        ...     task="driver-top3",
        ...     budget_percentage=0.05,
        ...     method="all",
        ...     num_runs=10,
        ...     seed=42
        ... )
        >>> 
        >>> # Print aggregated results
        >>> for method, stats in results.items():
        ...     if 'avg_actual_y_perf_of_selected' in stats:
        ...         print(f"{method}: {stats['avg_actual_y_perf_of_selected']:.4f}")
        
        >>> # Run specific methods with custom configuration
        >>> results = run_benchmark(
        ...     dataset="rel-avito",
        ...     task="user-ad-visit",
        ...     budget_percentage=0.10,
        ...     method=["ea", "greedy", "rl"],
        ...     num_runs=5,
        ...     tag="custom_experiment",
        ...     cache_dir="/custom/cache",
        ...     result_dir="/custom/results"
        ... )
        
        >>> # Run quick test with single method
        >>> results = run_benchmark(
        ...     method="all",
        ...     num_runs=10,
        ...     budget_percentage=0.05
        ... )
        
    Note:
        - Results are automatically saved to CSV files for further analysis
        - Performance trajectories are exported for visualization
        - All intermediate data is cached to speed up repeated experiments
        - Large datasets may require significant memory and storage
    """
    
    if isinstance(method, str):
        method = [method]
    
    default_params = {
        "tag": "hf",
        "cache_dir": "~/.cache/relbench_examples",
        "result_dir": "./results",
    }
    
    params = {**default_params, **kwargs}
    
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(
        dataset=dataset,
        task=task,
        budget_percentage=budget_percentage,
        method=method,
        num_runs=num_runs,
        seed=seed,
        **params
    )
    
    results = benchmark_main(args)
    
    return results 