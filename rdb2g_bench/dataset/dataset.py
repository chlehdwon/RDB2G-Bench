import os
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Union
from datasets import load_dataset

from .bench import RDB2GBench


def load_rdb2g_bench(
    result_dir: str = "./results",
    download: bool = True,
    cache_dir: Optional[str] = None,
    tag: str = "hf"
) -> RDB2GBench:
    """
    Load RDB2G-Bench results from local directory, download if missing.
    
    Args:
        result_dir: Directory containing the results
        download: Whether to download if results are missing
        cache_dir: Cache directory for Hugging Face datasets
        tag: Tag to use for downloaded data organization
        
    Returns:
        RDB2GBench object for accessing results
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
    
    Args:
        result_dir: Directory to save the organized results
        cache_dir: Cache directory for Hugging Face datasets
        dataset_names: List of specific datasets to download (if None, download all)
        task_names: List of specific tasks to download (if None, download all)
        tag: Tag to identify the download
        
    Returns:
        Dictionary mapping dataset/task combinations to saved file paths
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
    List all available datasets and tasks in the RDB2G-Bench dataset.
    
    Args:
        cache_dir: Cache directory for Hugging Face datasets
        
    Returns:
        DataFrame with unique dataset/task combinations and their statistics
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
