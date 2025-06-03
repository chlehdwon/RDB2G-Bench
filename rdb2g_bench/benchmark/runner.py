"""
Benchmark runner module - provides a functional interface to the benchmark system.
"""

import os
from typing import Dict, List, Union, Optional

from .benchmark import main as benchmark_main


def run_benchmark(
    dataset: str = "rel-f1",
    task: str = "driver-top3",
    budget_percentage: float = 0.05,
    method: Union[str, List[str]] = "all",
    num_runs: int = 10,
    seed: int = 0,
    **kwargs
) -> Dict:
    """
    Run benchmark analysis on the specified dataset and task.
    
    Args:
        dataset: Name of the RelBench dataset
        task: Name of the RelBench task  
        budget_percentage: Budget percentage for the search algorithm
        method: Which analysis method(s) to run ('all', 'ea', 'greedy', 'rl', 'bo')
        num_runs: Number of times to repeat the training and evaluation process
        seed: Initial random seed (seed for run i will be seed + i)
        **kwargs: Additional configuration parameters
        
    Available kwargs (with defaults):
        tag (str): Tag identifying the results sub-directory (default: "final")
        cache_dir (str): Directory for caching (default: "~/.cache/relbench_examples")
        result_dir (str): Root directory for results (default: "./results")
        
    Returns:
        Dictionary containing benchmark results
    """
    
    # Convert method to list if it's a string
    if isinstance(method, str):
        method = [method]
    
    # Set default values for benchmark parameters
    default_params = {
        "tag": "final",
        "cache_dir": os.path.expanduser("~/.cache/relbench_examples"),
        "result_dir": os.path.expanduser("./results"),
    }
    
    # Update defaults with provided kwargs
    params = {**default_params, **kwargs}
    
    # Create a mock args object to match the original interface
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
    
    # Run the benchmark
    results = benchmark_main(args)
    
    return results 