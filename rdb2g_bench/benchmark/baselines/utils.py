import torch
from typing import Optional, Tuple
from ..dataset import PerformancePredictionDataset
import numpy as np

def calculate_overall_rank(selected_actual_y: float,
                           overall_actual_y: torch.Tensor,
                           higher_is_better: bool):
    """
    Calculates the rank (position and percentile) of a selected performance.
    
    Args:
        selected_actual_y (float): The performance score to rank
        overall_actual_y (torch.Tensor): Tensor containing all performance values
        higher_is_better (bool): Whether higher values indicate better performance
        
    Returns:
        Optional[Dict]: Dictionary containing rank information with keys:
        
        rank_position_overall (int): Position in the overall ranking
        percentile_overall (float): Percentile ranking (0-100)
        total_samples_overall (int): Total number of samples in ranking
            
        Returns None if overall_actual_y is empty.
    """
    if overall_actual_y.numel() == 0:
        print("Warning: Overall actual y tensor is empty, cannot calculate overall rank.")
        return None

    overall_actual_y = overall_actual_y.squeeze()
    if overall_actual_y.ndim == 0:
        overall_actual_y = overall_actual_y.unsqueeze(0)

    num_total_overall = overall_actual_y.numel()
    if higher_is_better:
        num_better_or_equal = torch.sum(overall_actual_y >= selected_actual_y).item()
    else:
        num_better_or_equal = torch.sum(overall_actual_y <= selected_actual_y).item()

    rank_position = num_better_or_equal
    percentile = (rank_position / num_total_overall) * 100 if num_total_overall > 0 else 0

    return {
        "rank_position_overall": rank_position,
        "percentile_overall": percentile,
        "total_samples_overall": num_total_overall
    }

def get_performance_for_index(
    index: int,
    dataset: PerformancePredictionDataset,
    performance_cache: dict
) -> float:
    """
    Retrieves the performance for a given graph index, using a cache if available.
    
    Args:
        index (int): Graph index to retrieve performance for
        dataset (PerformancePredictionDataset): Dataset containing performance data
        performance_cache (dict): Cache for storing retrieved performance values
        
    Returns:
        Optional[float]: Performance value for the given index, or None if not found
            or if performance data is invalid (NaN/None).
    """
    if index in performance_cache:
        return performance_cache[index]

    try:
        row = dataset.df_result_group[dataset.df_result_group['idx'] == index]
        if len(row) == 0:
            print(f"Warning: No performance data found for index {index}")
            return None
        
        performance = row.iloc[0][dataset.target_col]
        
        if performance is None or (isinstance(performance, float) and np.isnan(performance)):
            print(f"Warning: NaN or None performance found for index {index}")
            return None
            
        return performance
        
    except Exception as e:
        print(f"Error retrieving performance for index {index}: {str(e)}")
        return None

def update_trajectory_and_best(
    index: int,
    perf: Optional[float],
    performance_cache: dict,
    initial_cache_size: int,
    total_evaluated_count: int,
    performance_trajectory: list,
    global_best_perf: float,
    global_best_index: int,
    higher_is_better: bool,
) -> Tuple[int, float, int]:
    """
    Updates trajectory and global best after a performance evaluation.
    
    Args:
        index (int): Index of the evaluated architecture
        perf (Optional[float]): Performance value obtained
        performance_cache (dict): Cache storing performance values
        initial_cache_size (int): Initial size of performance cache
        total_evaluated_count (int): Current count of evaluated architectures
        performance_trajectory (list): List tracking performance over time
        global_best_perf (float): Current best performance found
        global_best_index (int): Index of current best architecture
        higher_is_better (bool): Whether higher values indicate better performance
        
    Returns:
        Tuple[int, float, int]: Updated values:
        
        new_total_evaluated_count (int): Updated evaluation count
        new_global_best_perf (float): Updated best performance
        new_global_best_index (int): Updated best architecture index
    """
    new_total_evaluated_count = total_evaluated_count
    new_global_best_perf = global_best_perf
    new_global_best_index = global_best_index

    if perf is not None and np.isfinite(perf):
        if len(performance_cache) > initial_cache_size:
            new_total_evaluated_count += 1

            is_better = False
            if new_global_best_index == -1:
                is_better = True
            elif higher_is_better and perf > new_global_best_perf:
                is_better = True
            elif not higher_is_better and perf < new_global_best_perf:
                is_better = True

            if is_better:
                new_global_best_perf = perf
                new_global_best_index = index

            performance_trajectory.append((new_total_evaluated_count, new_global_best_perf))

        else:
            is_better_than_current_global = False
            if new_global_best_index == -1:
                 is_better_than_current_global = True
            elif higher_is_better and perf > new_global_best_perf:
                 is_better_than_current_global = True
            elif not higher_is_better and perf < new_global_best_perf:
                 is_better_than_current_global = True

            if is_better_than_current_global:
                new_global_best_perf = perf
                new_global_best_index = index

    return new_total_evaluated_count, new_global_best_perf, new_global_best_index

def pad_trajectory(
    performance_trajectory: list,
    total_evaluated_count: int,
    evaluation_budget: int,
    method_name: str = "Search"
) -> None:
    """Pads the performance trajectory if the search ended before using the full budget."""
    if total_evaluated_count < evaluation_budget:
        if performance_trajectory:
            final_best_perf_to_pad = performance_trajectory[-1][1]
            if np.isfinite(final_best_perf_to_pad):
                print(f"Padding {method_name} trajectory from {total_evaluated_count + 1} to {evaluation_budget} with final best perf: {final_best_perf_to_pad:.4f}")
                for i in range(total_evaluated_count + 1, evaluation_budget + 1):
                    performance_trajectory.append((i, final_best_perf_to_pad))
            else:
                print(f"Warning: {method_name} Search ended early ({total_evaluated_count}/{evaluation_budget}) but last best performance is not finite ({final_best_perf_to_pad}). Cannot pad trajectory.")
        else:
            print(f"Warning: {method_name} Search ended early ({total_evaluated_count}/{evaluation_budget}) but performance trajectory is empty. Cannot pad trajectory.")

def calculate_evaluation_time(
    index: int,
    dataset: PerformancePredictionDataset,
    time_cache: dict
) -> float:
    """
    Calculates the evaluation time: eval_time = train_time * 20 + valid_time + test_time.
    
    Args:
        index (int): Graph index to calculate evaluation time for
        dataset (PerformancePredictionDataset): Dataset containing timing data
        time_cache (dict): Cache for storing calculated evaluation times
        
    Returns:
        Optional[float]: Total evaluation time in seconds, or None if timing data
            is not available or invalid.
    """
    epochs = 20

    if index in time_cache:
        return time_cache[index]
        
    try:
        row = dataset.df_result_group[dataset.df_result_group['idx'] == index]
        if len(row) == 0:
            print(f"Warning: No time data found for index {index}")
            return None
            
        train_time = row.iloc[0]['train_time']
        
        if train_time is None or (isinstance(train_time, float) and np.isnan(train_time)):
            print(f"Warning: NaN or None train_time found for index {index}")
            return None
        
        valid_time = row.iloc[0]['valid_time']
        test_time = row.iloc[0]['test_time']
        
        if (valid_time is None or (isinstance(valid_time, float) and np.isnan(valid_time)) or
            test_time is None or (isinstance(test_time, float) and np.isnan(test_time))):
            print(f"Warning: NaN or None valid_time or test_time found for index {index}")
            eval_time = train_time * epochs + train_time + train_time
        else:
            eval_time = (train_time + valid_time) * epochs + test_time
        
        time_cache[index] = eval_time
        
        return eval_time
        
    except Exception as e:
        print(f"Error calculating evaluation time for index {index}: {str(e)}")
        return None