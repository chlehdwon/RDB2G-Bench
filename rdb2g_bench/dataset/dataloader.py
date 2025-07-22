import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
import numpy as np


class RDB2GBench:
    """
    Main benchmark data container for RDB2G-Bench results.
    
    This class loads and organizes all benchmark data from a specified directory
    structure and provides hierarchical access to results through dataset and task names.
    
    The expected directory structure is::
    
        results/
        └── tables/
            └── dataset_name/
                └── task_name/
                    └── tag/
                        └── gnn_name/
                            ├── 0.csv
                            ├── 1.csv
                            └── ...
    
    Attributes:
        result_dir (Path): Path to the results directory
        data (Dict): Nested dictionary containing loaded data organized by dataset/task
        
    Example:
        >>> bench = RDB2GBench("./results")
        >>> available = bench.get_available()
        >>> dataset_accessor = bench['rel-f1']
    """
    
    def __init__(self, result_dir: Union[str, Path]):
        """
        Initialize RDB2GBench with results from specified directory.
        
        Args:
            result_dir (Union[str, Path]): Path to directory containing benchmark results
        """
        self.result_dir = Path(result_dir)
        self.data = self._load_all_data()
    
    def _load_all_data(self) -> Dict:
        """
        Load all CSV files and organize by dataset/task.
        
        Scans the directory structure for CSV files containing benchmark results
        and loads them into a nested dictionary structure for fast access.
        
        Returns:
            Dict: Nested dictionary with structure {dataset: {task: DataFrame}}
        """
        data = {}
        tables_dir = self.result_dir / "tables"
        
        if not tables_dir.exists():
            return data
        
        # Iterate through dataset directories
        for dataset_dir in tables_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            dataset_name = dataset_dir.name
            data[dataset_name] = {}
            
            # Iterate through task directories
            for task_dir in dataset_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                    
                task_name = task_dir.name
                
                # Find CSV files (look in tag/gnn subdirectories)
                csv_files = []
                for tag_dir in task_dir.iterdir():
                    if tag_dir.is_dir():
                        # Look for GNN subdirectories within tag directory
                        for gnn_dir in tag_dir.iterdir():
                            if gnn_dir.is_dir():
                                csv_files.extend(gnn_dir.glob("*.csv"))
                
                if csv_files:
                    # Load and combine all CSV files
                    dfs = []
                    for csv_file in csv_files:
                        df = pd.read_csv(csv_file)
                        df['seed'] = int(csv_file.stem)  # Add seed from filename
                        # Add GNN info from directory name
                        gnn_name = csv_file.parent.name
                        if 'gnn' not in df.columns:
                            df['gnn'] = gnn_name
                        dfs.append(df)
                    
                    combined_df = pd.concat(dfs, ignore_index=True)
                    data[dataset_name][task_name] = combined_df
        
        return data
    
    def get_available(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get available datasets, tasks, and GNN models.
        
        Returns:
            Dict[str, Dict[str, List[str]]]: Nested dictionary mapping dataset names to task names to lists of available GNN models
            
        Example:
            >>> bench = RDB2GBench("./results")
            >>> available = bench.get_available()
            >>> print(available)
            {'rel-f1': {'driver-top3': ['GIN', 'GPS', 'GraphSAGE'], ...}, ...}
        """
        result = {}
        for dataset, tasks in self.data.items():
            result[dataset] = {}
            for task, task_data in tasks.items():
                if 'gnn' in task_data.columns:
                    result[dataset][task] = sorted(task_data['gnn'].unique().tolist())
                else:
                    result[dataset][task] = []
        return result
    
    def __getitem__(self, dataset_name: str):
        """
        Access dataset by name.
        
        Args:
            dataset_name (str): Name of the dataset to access
            
        Returns:
            DatasetAccessor: Accessor object for the specified dataset
            
        Raises:
            KeyError: If dataset is not found
        """
        if dataset_name not in self.data:
            raise KeyError(f"Dataset '{dataset_name}' not found")
        return DatasetAccessor(self.data[dataset_name], dataset_name)


class DatasetAccessor:
    """
    Accessor for tasks within a specific dataset.
    
    This class provides access to all tasks available for a particular dataset
    and returns TaskAccessor objects for further navigation.
    
    Attributes:
        tasks_data (Dict): Dictionary containing task data for this dataset
        dataset_name (str): Name of the dataset being accessed
    """
    
    def __init__(self, tasks_data: Dict, dataset_name: str):
        """
        Initialize DatasetAccessor.
        
        Args:
            tasks_data (Dict): Dictionary containing task data
            dataset_name (str): Name of the dataset
        """
        self.tasks_data = tasks_data
        self.dataset_name = dataset_name
    
    def __getitem__(self, task_name: str):
        """
        Access task by name within this dataset.
        
        Args:
            task_name (str): Name of the task to access
            
        Returns:
            TaskAccessor: Accessor object for the specified task
            
        Raises:
            KeyError: If task is not found in this dataset
        """
        if task_name not in self.tasks_data:
            raise KeyError(f"Task '{task_name}' not found")
        return TaskAccessor(self.tasks_data[task_name], self.dataset_name, task_name)


class TaskAccessor:
    """
    Accessor for GNN models within a specific task.
    
    This class provides access to all GNN models available for a 
    particular dataset/task combination and returns GNNAccessor objects
    for further navigation.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing all results for this task
        dataset_name (str): Name of the dataset
        task_name (str): Name of the task
    """
    
    def __init__(self, data: pd.DataFrame, dataset_name: str, task_name: str):
        """
        Initialize TaskAccessor.
        
        Args:
            data (pd.DataFrame): DataFrame containing task results
            dataset_name (str): Name of the dataset
            task_name (str): Name of the task
        """
        self.data = data
        self.dataset_name = dataset_name
        self.task_name = task_name
    
    def __getitem__(self, gnn_name: str):
        """
        Access specific GNN model by name.
        
        Args:
            gnn_name (str): Name of the GNN model to access
            
        Returns:
            GNNAccessor: Accessor object for the specified GNN model
            
        Raises:
            KeyError: If GNN model is not found in this task
        """
        gnn_data = self.data[self.data['gnn'] == gnn_name]
        if gnn_data.empty:
            raise KeyError(f"GNN '{gnn_name}' not found")
        return GNNAccessor(gnn_data, self.dataset_name, self.task_name, gnn_name)
    
    def get_available_gnns(self) -> List[str]:
        """
        Get list of available GNN models.
        
        Returns:
            List[str]: Sorted list of available GNN models for this task
            
        Example:
            >>> task = bench['rel-f1']['driver-top3']
            >>> gnns = task.get_available_gnns()
            >>> print(gnns)
            ['GIN', 'GPS', 'GraphSAGE']
        """
        return sorted(self.data['gnn'].unique().tolist())
    
    def get_available_indices(self) -> List[int]:
        """
        Get list of available graph configuration indices across all GNNs.
        
        Returns:
            List[int]: Sorted list of available indices for this task
            
        Example:
            >>> task = bench['rel-f1']['driver-top3']
            >>> indices = task.get_available_indices()
            >>> print(indices)
            [0, 1, 2, 3, ..., 49]
        """
        return sorted(self.data['idx'].unique().tolist())


class GNNAccessor:
    """
    Accessor for graph configurations (indices) within a specific GNN model.
    
    This class provides access to all graph configurations available for a 
    particular dataset/task/GNN combination and returns IndexAccessor objects
    containing aggregated results.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing all results for this GNN model
        dataset_name (str): Name of the dataset
        task_name (str): Name of the task
        gnn_name (str): Name of the GNN model
    """
    
    def __init__(self, data: pd.DataFrame, dataset_name: str, task_name: str, gnn_name: str):
        """
        Initialize GNNAccessor.
        
        Args:
            data (pd.DataFrame): DataFrame containing GNN model results
            dataset_name (str): Name of the dataset
            task_name (str): Name of the task
            gnn_name (str): Name of the GNN model
        """
        self.data = data
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.gnn_name = gnn_name
    
    def __getitem__(self, idx: int):
        """
        Access specific graph configuration by index.
        
        Args:
            idx (int): Index of the graph configuration to access
            
        Returns:
            IndexAccessor: Accessor object for the specified graph configuration
            
        Raises:
            KeyError: If index is not found for this GNN model
        """
        idx_data = self.data[self.data['idx'] == idx]
        if idx_data.empty:
            raise KeyError(f"Index {idx} not found for GNN '{self.gnn_name}'")
        return IndexAccessor(idx_data, self.dataset_name, self.task_name, self.gnn_name, idx)
    
    def get_available_indices(self) -> List[int]:
        """
        Get list of available graph configuration indices for this GNN model.
        
        Returns:
            List[int]: Sorted list of available indices for this GNN model
            
        Example:
            >>> gnn = bench['rel-f1']['driver-top3']['GraphSAGE']
            >>> indices = gnn.get_available_indices()
            >>> print(indices)
            [0, 1, 2, 3, ..., 49]
        """
        return sorted(self.data['idx'].unique().tolist())


class IndexAccessor:
    """
    Accessor for aggregated results of a specific graph configuration.
    
    This class computes and provides access to aggregated statistics across
    multiple random seeds for a specific graph configuration. It automatically
    calculates means and standard deviations for performance metrics.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing results for this configuration
        dataset_name (str): Name of the dataset
        task_name (str): Name of the task
        gnn_name (str): Name of the GNN model
        idx (int): Index of the graph configuration
        _stats (Optional[Dict]): Cached computed statistics
    """
    
    def __init__(self, data: pd.DataFrame, dataset_name: str, task_name: str, gnn_name: str, idx: int):
        """
        Initialize IndexAccessor.
        
        Args:
            data (pd.DataFrame): DataFrame containing results for this index
            dataset_name (str): Name of the dataset
            task_name (str): Name of the task
            gnn_name (str): Name of the GNN model
            idx (int): Index of the graph configuration
        """
        self.data = data
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.gnn_name = gnn_name
        self.idx = idx
        self._stats = None
    
    @property
    def stats(self):
        """
        Compute and cache aggregated statistics for this graph configuration.
        
        Calculates mean and standard deviation for performance metrics across
        all seeds, and mean values for parameter counts and timing information.
        
        Returns a dictionary containing computed statistics with keys:
        
        - params: Mean parameter count (int)
        - \\*_time: Mean timing values (float) for train/valid/test
        - \\*_mean: Mean values for other numeric columns (float)
        - \\*_std: Standard deviation for other numeric columns (float)
        
        Returns:
            Dict[str, Union[int, float]]
        """
        if self._stats is None:
            self._stats = {}
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col not in ['idx', 'seed']:
                    values = self.data[col].dropna()
                    if len(values) > 0:
                        if col == 'params':
                            # Parameter count as integer mean
                            self._stats['params'] = int(values.mean())
                        elif col.endswith('_time'):
                            # Timing metrics as float mean
                            time_name = col
                            self._stats[time_name] = float(values.mean())
                        else:
                            # Other metrics with mean and std
                            self._stats[f"{col}_mean"] = float(values.mean())
                            self._stats[f"{col}_std"] = float(values.std()) if len(values) > 1 else 0.0
        return self._stats
    
    def __getitem__(self, column_name: str) -> float:
        """
        Get specific statistic by column name.
        
        Supports accessing statistics with or without '_mean' suffix.
        
        Args:
            column_name (str): Name of the statistic to retrieve
            
        Returns:
            float: Value of the requested statistic
            
        Raises:
            KeyError: If column is not found in computed statistics
            
        Example:
            >>> result = bench['rel-f1']['driver-top3'][0]
            >>> test_perf = result['test_metric']  # Gets test_metric_mean
            >>> test_mean = result['test_metric_mean']  # Same as above
            >>> test_std = result['test_metric_std']
        """
        if column_name in self.stats:
            return self.stats[column_name]
        
        if not column_name.endswith('_mean') and not column_name.endswith('_std'):
            mean_key = f"{column_name}_mean"
            if mean_key in self.stats:
                return self.stats[mean_key]
        
        raise KeyError(f"Column '{column_name}' not found")
    
    def __str__(self) -> str:
        """
        String representation of this graph configuration's results.
        
        Returns:
            str: Formatted string showing index and computed statistics
        """
        return f"Index {self.idx}: {self.stats}" 