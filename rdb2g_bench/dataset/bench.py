import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
import numpy as np


class RDB2GBench:    
    def __init__(self, result_dir: Union[str, Path]):
        self.result_dir = Path(result_dir)
        self.data = self._load_all_data()
    
    def _load_all_data(self) -> Dict:
        """Load all CSV files and organize by dataset/task."""
        data = {}
        tables_dir = self.result_dir / "tables"
        
        if not tables_dir.exists():
            return data
        
        for dataset_dir in tables_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            dataset_name = dataset_dir.name
            data[dataset_name] = {}
            
            for task_dir in dataset_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                    
                task_name = task_dir.name
                
                # Find CSV files (look in tag subdirectories)
                csv_files = []
                for tag_dir in task_dir.iterdir():
                    if tag_dir.is_dir():
                        csv_files.extend(tag_dir.glob("*.csv"))
                
                if csv_files:
                    # Load and combine all CSV files
                    dfs = []
                    for csv_file in csv_files:
                        df = pd.read_csv(csv_file)
                        df['seed'] = int(csv_file.stem)
                        dfs.append(df)
                    
                    combined_df = pd.concat(dfs, ignore_index=True)
                    data[dataset_name][task_name] = combined_df
        
        return data
    
    def get_available(self) -> Dict[str, List[str]]:
        """Get available datasets and tasks."""
        return {dataset: list(tasks.keys()) for dataset, tasks in self.data.items()}
    
    def __getitem__(self, dataset_name: str):
        if dataset_name not in self.data:
            raise KeyError(f"Dataset '{dataset_name}' not found")
        return DatasetAccessor(self.data[dataset_name], dataset_name)


class DatasetAccessor:
    """Simple accessor for dataset tasks."""
    def __init__(self, tasks_data: Dict, dataset_name: str):
        self.tasks_data = tasks_data
        self.dataset_name = dataset_name
    
    def __getitem__(self, task_name: str):
        if task_name not in self.tasks_data:
            raise KeyError(f"Task '{task_name}' not found")
        return TaskAccessor(self.tasks_data[task_name], self.dataset_name, task_name)


class TaskAccessor:
    """Simple accessor for task results."""
    
    def __init__(self, data: pd.DataFrame, dataset_name: str, task_name: str):
        self.data = data
        self.dataset_name = dataset_name
        self.task_name = task_name
    
    def __getitem__(self, idx: int):
        idx_data = self.data[self.data['idx'] == idx]
        if idx_data.empty:
            raise KeyError(f"Index {idx} not found")
        return IndexAccessor(idx_data, self.dataset_name, self.task_name, idx)
    
    def get_available_indices(self) -> List[int]:
        return sorted(self.data['idx'].unique().tolist())


class IndexAccessor:    
    def __init__(self, data: pd.DataFrame, dataset_name: str, task_name: str, idx: int):
        self.data = data
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.idx = idx
        self._stats = None
    
    @property
    def stats(self):
        if self._stats is None:
            self._stats = {}
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col not in ['idx', 'seed']:
                    values = self.data[col].dropna()
                    if len(values) > 0:
                        if col == 'params':
                            self._stats['params'] = int(values.mean())
                        elif col.endswith('_time'):
                            time_name = col
                            self._stats[time_name] = float(values.mean())
                        else:
                            self._stats[f"{col}_mean"] = float(values.mean())
                            self._stats[f"{col}_std"] = float(values.std()) if len(values) > 1 else 0.0
        return self._stats
    
    def __getitem__(self, column_name: str) -> float:
        if column_name in self.stats:
            return self.stats[column_name]
        
        if not column_name.endswith('_mean') and not column_name.endswith('_std'):
            mean_key = f"{column_name}_mean"
            if mean_key in self.stats:
                return self.stats[mean_key]
        
        raise KeyError(f"Column '{column_name}' not found")
    
    def __str__(self) -> str:
        return f"Index {self.idx}: {self.stats}" 