import os
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Union
from datasets import load_dataset

from .dataloader import RDB2GBench


def load_rdb2g_bench(
    result_dir: str = "./results",
    download: bool = True,
    cache_dir: Optional[str] = None,
    tag: str = "hf"
) -> RDB2GBench:
    """
    Load RDB2G-Bench results from local directory, download if missing.
    
    This function serves as the main entry point for accessing RDB2G-Bench data.
    It first checks for existing data in the specified directory, and if no data
    is found and download is enabled, it automatically downloads the dataset
    from Hugging Face Hub.
    
    Args:
        result_dir (str): Directory containing the benchmark results. 
            Defaults to "./results".
        download (bool): Whether to download data if results are missing. 
            Defaults to True.
        cache_dir (Optional[str]): Cache directory for Hugging Face datasets.
            If None, uses default HF cache location.
        tag (str): Tag to use for downloaded data organization. 
            Defaults to "hf".
        
    Returns:
        RDB2GBench: Benchmark object for accessing organized results with
            hierarchical access pattern: bench[dataset][task][idx].
        
    Raises:
        RuntimeError: If no valid data is found in the result directory
            after download attempts.
            
    Example:
        >>> bench = load_rdb2g_bench("./my_results")
        >>> available = bench.get_available()
        >>> result = bench['rel-f1']['driver-top3'][0]
    """
    result_path = Path(result_dir)
    tables_path = result_path / "tables"
    
    has_data = False
    if tables_path.exists():
        for dataset_dir in tables_path.iterdir():
            if dataset_dir.is_dir():
                for task_dir in dataset_dir.iterdir():
                    if task_dir.is_dir():
                        for tag_dir in task_dir.iterdir():
                            if tag_dir.is_dir() and list(tag_dir.glob("*.csv")):
                                has_data = True
                                break
                    if has_data:
                        break
            if has_data:
                break
    
    if not has_data and download:
        print(f"No data found in {result_dir}, downloading from Hugging Face...")
        download_rdb2g_bench(
            result_dir=result_dir,
            cache_dir=cache_dir,
            tag=tag
        )
    
    bench = RDB2GBench(result_dir)
    
    if not bench.get_available():
        raise RuntimeError(f"No valid data found in {result_dir}")
    
    return bench


def download_rdb2g_bench(
    result_dir: str = "./results",
    cache_dir: Optional[str] = None,
    dataset_names: Optional[List[str]] = None,
    task_names: Optional[List[str]] = None,
    tag: str = "hf",
) -> Dict[str, List[str]]:
    """
    Download RDB2G-Bench dataset from Hugging Face and organize it by dataset/task.
    
    This function downloads the complete or filtered RDB2G-Bench dataset from
    Hugging Face Hub and organizes it into a structured directory format.
    The data is grouped by dataset and task, with separate CSV files for each
    random seed.
    
    Args:
        result_dir (str): Directory to save the organized results.
            Will be created if it doesn't exist. Defaults to "./results".
        cache_dir (Optional[str]): Cache directory for Hugging Face datasets.
            If None, uses default HF cache location.
        dataset_names (Optional[List[str]]): List of specific datasets to download.
            If None, downloads all available datasets.
        task_names (Optional[List[str]]): List of specific tasks to download.
            If None, downloads all available tasks.
        tag (str): Tag to identify the download and organize files.
            Defaults to "hf".
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping dataset/task combinations to lists of saved file paths.
        
        Keys are in format "dataset/task".
            
    Example:
        >>> saved_files = download_rdb2g_bench(
        ...     dataset_names=['rel-f1'],
        ...     task_names=['driver-top3']
        ... )
        >>> print(saved_files)
        {'rel-f1/driver-top3': ['./results/tables/rel-f1/driver-top3/hf/0.csv', ...]}
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(
        "kaistdata/RDB2G-Bench",
        cache_dir=cache_dir,
        split="train"
    )
    
    df = dataset.to_pandas()
    
    if dataset_names is not None:
        df = df[df['dataset'].isin(dataset_names)]
        
    if task_names is not None:
        df = df[df['task'].isin(task_names)]
    
    if df.empty:
        return {}
    
    saved_files = {}
    grouped = df.groupby(['dataset', 'task'])
    
    for (dataset_name, task_name), group_df in grouped:
        task_dir = result_dir / "tables" / dataset_name / task_name / tag
        task_dir.mkdir(parents=True, exist_ok=True)
        
        group_df = group_df.sort_values(['seed', 'idx'])
        
        seed_files = []
        for seed, seed_df in group_df.groupby('seed'):
            output_df = seed_df[[
                'idx', 'graph', 'test_metric', 'params', 
                'train_time', 'valid_time', 'test_time'
            ]].copy()
            
            filename = f"{seed}.csv"
            filepath = task_dir / filename
            output_df.to_csv(filepath, index=False)
            seed_files.append(str(filepath))
        
        combination_key = f"{dataset_name}/{task_name}"
        saved_files[combination_key] = seed_files
    
    return saved_files


def get_dataset_stats(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Get comprehensive statistics about all available datasets and tasks in RDB2G-Bench.
    
    This function loads the complete dataset from Hugging Face and computes
    aggregate statistics for each dataset/task combination, including counts
    of unique indices and seeds, as well as performance metrics statistics.
    
    Args:
        cache_dir (Optional[str]): Cache directory for Hugging Face datasets.
            If None, uses default HF cache location.
        
    Returns:
        pd.DataFrame: DataFrame with statistical information about all datasets and tasks.
        
        - dataset: Dataset name
        - task: Task name  
        - idx: Number of unique graph configurations
        - seed: Number of random seeds
        - test_metric_mean: Mean test performance
        - test_metric_std: Standard deviation of test performance
        - test_metric_min: Minimum test performance
        - test_metric_max: Maximum test performance
            
    Example:
        >>> stats = get_dataset_stats()
        >>> print(stats.head())
        dataset    task         idx  seed  test_metric_mean  test_metric_std  ...
        rel-f1     driver-top3   50    10            0.8542           0.0123  ...
    """
    dataset = load_dataset(
        "kaistdata/RDB2G-Bench",
        cache_dir=cache_dir,
        split="train"
    )
    
    df = dataset.to_pandas()
    
    stats = df.groupby(['dataset', 'task']).agg({
        'idx': 'nunique',
        'seed': 'nunique',
        'test_metric': ['mean', 'std', 'min', 'max']
    }).round(4)

    stats.columns = ['_'.join(col).strip() for col in stats.columns]

    stats = stats.rename(columns={
        'idx_nunique': 'idx',
        'seed_nunique': 'seed',
    })
    stats = stats.reset_index()
    
    return stats
