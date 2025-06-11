import torch
import numpy as np
import random
import time
from typing import Tuple, Dict, Union, List, Optional

from ..dataset import PerformancePredictionDataset
from ..micro_action import MicroActionSet
from .utils import calculate_overall_rank, get_performance_for_index, update_trajectory_and_best, pad_trajectory, calculate_evaluation_time

def forward_greedy_heuristic_analysis(
    dataset: PerformancePredictionDataset,
    micro_action_set: MicroActionSet,
    overall_actual_y: torch.Tensor,
    higher_is_better: bool,
    termination_threshold_ratio: float,
    method_name: str = "Forward Greedy Heuristic",
):
    """
    Perform Neural Architecture Search using Forward Greedy Strategy.
    
    This function implements forward greedy search that starts from minimal graphs
    and progressively adds edges through micro actions. It explores the architecture
    space by systematically expanding configurations to improve performance.
    
    Args:
        dataset (PerformancePredictionDataset): Dataset containing architecture 
            performance data
        micro_action_set (MicroActionSet): Set of micro actions for architecture
            space exploration
        overall_actual_y (torch.Tensor): Complete performance tensor for
            ranking calculations
        higher_is_better (bool): Whether higher performance values are better
        termination_threshold_ratio (float): Fraction of total architectures to
            evaluate as budget
        method_name (str): Name identifier for this method. 
            Defaults to "Forward Greedy Heuristic".
            
    Returns:
        Dict[str, Union[str, int, float, List, Optional[int]]]: Dictionary containing search results and performance metrics.
        
        - method (str): Method name
        - selected_graph_id (Optional[int]): Index of best found architecture
        - actual_y_perf_of_selected (float): Performance of selected architecture
        - selection_metric_value (float): Metric value used for selection
        - selected_graph_origin (str): Origin method name
        - discovered_count (int): Number of architectures evaluated
        - total_iterations_run (int): Number of greedy steps completed
        - rank_position_overall (float): Rank among all architectures
        - percentile_overall (float): Percentile ranking
        - total_samples_overall (int): Total available architectures
        - performance_trajectory (List): Performance over time
        - total_evaluation_time (float): Time spent on evaluations
        - total_run_time (float): Total algorithm runtime
            
    Example:
        >>> results = forward_greedy_heuristic_analysis(
        ...     dataset=dataset,
        ...     micro_action_set=micro_actions,
        ...     overall_actual_y=y_tensor,
        ...     higher_is_better=True,
        ...     termination_threshold_ratio=0.05
        ... )
        >>> print(f"Best architecture: {results['selected_graph_id']}")
        >>> print(f"Performance: {results['actual_y_perf_of_selected']:.4f}")
    """
    performance_cache = {}
    time_cache = {}
    performance_trajectory = []
    total_evaluated_count = 0
    num_total_valid_graphs = len(micro_action_set.valid_edge_sets_list)
    discovered_count = 0
    global_best_perf = float('-inf') if higher_is_better else float('inf')
    global_best_index = -1
    current_index = -1
    final_iteration_count = 0
    method_origin = "Forward Greedy"
    total_samples = overall_actual_y.numel() if overall_actual_y is not None else 0
    total_evaluation_time = 0.0
    
    start_time = time.time()

    termination_count_threshold = int(termination_threshold_ratio * num_total_valid_graphs)
    if termination_count_threshold < 1: termination_count_threshold = 1
    print(f"{method_name}: Budget set to {termination_count_threshold} unique evaluations.")

    initial_candidate_indices = []
    if micro_action_set.valid_edge_sets_list:
        for idx, edge_set in enumerate(micro_action_set.valid_edge_sets_list):
            try:
                num_edges = sum(1 for edge_present in edge_set if edge_present)
                if num_edges == 1:
                     initial_candidate_indices.append(idx)
            except TypeError:
                print(f"Warning: Cannot process edge_set at index {idx}. Skipping.")
                continue

    best_initial_perf = float('-inf') if higher_is_better else float('inf')
    best_initial_idx = -1

    if initial_candidate_indices:
        print(f"Forward Greedy (Iter 1): Evaluating {len(initial_candidate_indices)} single-edge candidates...")
        final_iteration_count = 1

        random.shuffle(initial_candidate_indices)
        for idx in initial_candidate_indices:
            if total_evaluated_count >= termination_count_threshold:
                print(f"Forward Greedy (Iter 1): Budget reached ({total_evaluated_count}/{termination_count_threshold}) during initial eval.")
                break

            initial_cache_size = len(performance_cache)
            perf = get_performance_for_index(idx, dataset, performance_cache)
            
            if perf is not None:
                performance_cache[idx] = perf
                
                eval_time = calculate_evaluation_time(idx, dataset, time_cache)
                if eval_time is not None:
                    if len(performance_cache) > initial_cache_size:
                        total_evaluation_time += eval_time

            total_evaluated_count, global_best_perf, global_best_index = \
                update_trajectory_and_best(
                    idx, perf, performance_cache, initial_cache_size,
                    total_evaluated_count, performance_trajectory,
                    global_best_perf, global_best_index, higher_is_better
                )

            if perf is not None:
                if higher_is_better:
                    if perf > best_initial_perf:
                        best_initial_perf = perf
                        best_initial_idx = idx
                else:
                    if perf < best_initial_perf:
                        best_initial_perf = perf
                        best_initial_idx = idx

            if perf is not None and len(performance_cache) > initial_cache_size:
                 discovered_count += 1

        if best_initial_idx != -1:
            current_index = best_initial_idx
            print(f"Forward Greedy (Iter 1): Best initial candidate found: Index {current_index}, Perf: {best_initial_perf:.4f}. Total evaluated so far: {total_evaluated_count}")
        else:
            print(f"Warning: No valid single-edge candidate identified or budget exhausted during initial eval (total evaluated: {total_evaluated_count}).")

    if current_index == -1:
        print(f"Error: Could not establish a valid starting index. Cannot proceed with search loop.")
        final_selected_perf = performance_cache.get(global_best_index, np.nan) if global_best_index != -1 else np.nan
        results = {
            "method": method_name,
            "selected_graph_id": global_best_index if global_best_index !=-1 else None,
            "actual_y_perf_of_selected": final_selected_perf,
            "selection_metric_value": final_selected_perf,
            "selected_graph_origin": method_origin,
            "discovered_count": total_evaluated_count,
            "total_iterations_run": final_iteration_count,
            "rank_position_overall": np.nan,
            "percentile_overall": np.nan,
            "total_samples_overall": total_samples,
            "performance_trajectory": performance_trajectory,
            "total_evaluation_time": total_evaluation_time,
            "total_run_time": time.time() - start_time
        }
        pad_trajectory(performance_trajectory, total_evaluated_count, termination_count_threshold, method_name)
        if global_best_index != -1 and overall_actual_y is not None and not np.isnan(final_selected_perf):
            rank_info = calculate_overall_rank(final_selected_perf, overall_actual_y, higher_is_better)
            if rank_info:
                results["rank_position_overall"] = rank_info["rank_position_overall"]
                results["percentile_overall"] = rank_info["percentile_overall"]
        return results

    forward_actions = []
    action_names = ["add_fk_pk_edge", "convert_row_to_edge", "convert_edge_to_row"]
    for name in action_names:
        if hasattr(micro_action_set, name):
            forward_actions.append(getattr(micro_action_set, name))

    _, final_iteration_count, performance_cache, total_evaluated_count, performance_trajectory, final_global_best_perf, final_global_best_index, time_cache, total_evaluation_time = _perform_greedy_search(
        initial_index=current_index,
        action_functions=forward_actions,
        dataset=dataset,
        micro_action_set=micro_action_set,
        performance_cache=performance_cache,
        termination_count_threshold=termination_count_threshold,
        higher_is_better=higher_is_better,
        num_total_valid_graphs=num_total_valid_graphs,
        global_best_perf=global_best_perf,
        global_best_index=global_best_index,
        total_evaluated_count=total_evaluated_count,
        performance_trajectory=performance_trajectory,
        time_cache=time_cache,
        total_evaluation_time=total_evaluation_time,
        start_iteration=final_iteration_count,
        method_name_for_print="Forward Greedy"
    )

    pad_trajectory(performance_trajectory, total_evaluated_count, termination_count_threshold, method_name)

    total_run_time = time.time() - start_time

    final_selected_perf = performance_cache.get(final_global_best_index, np.nan) if final_global_best_index != -1 else np.nan
    results = {
        "method": method_name,
        "selected_graph_id": final_global_best_index if final_global_best_index != -1 else None,
        "actual_y_perf_of_selected": final_selected_perf,
        "selection_metric_value": final_selected_perf,
        "selected_graph_origin": method_origin,
        "discovered_count": total_evaluated_count,
        "total_iterations_run": final_iteration_count,
        "rank_position_overall": np.nan,
        "percentile_overall": np.nan,
        "total_samples_overall": total_samples,
        "performance_trajectory": performance_trajectory,
        "total_evaluation_time": total_evaluation_time,
        "total_run_time": total_run_time
    }

    if final_global_best_index != -1 and overall_actual_y is not None and not np.isnan(final_selected_perf):
        rank_info = calculate_overall_rank(
            final_selected_perf,
            overall_actual_y,
            higher_is_better
        )
        if rank_info:
            results["rank_position_overall"] = rank_info["rank_position_overall"]
            results["percentile_overall"] = rank_info["percentile_overall"]

    return results

def backward_greedy_heuristic_analysis(
    dataset: PerformancePredictionDataset,
    micro_action_set: MicroActionSet,
    overall_actual_y: torch.Tensor,
    higher_is_better: bool,
    termination_threshold_ratio: float,
    method_name: str = "Backward Greedy Heuristic",
):
    """
    Perform Neural Architecture Search using Backward Greedy Strategy.
    
    This function implements backward greedy search that starts from full graph
    configurations and iteratively removes edges or applies
    transformations to improve performance. The algorithm greedily selects the
    best local improvement at each step.
    
    Args:
        dataset (PerformancePredictionDataset): Dataset containing architecture 
            performance data
        micro_action_set (MicroActionSet): Set of micro actions for architecture
            space exploration
        overall_actual_y (torch.Tensor): Complete performance tensor for
            ranking calculations
        higher_is_better (bool): Whether higher performance values are better
        termination_threshold_ratio (float): Fraction of total architectures to
            evaluate as budget
        method_name (str): Name identifier for this method. 
            Defaults to "Backward Greedy Heuristic".
            
    Returns:
        Dict[str, Union[str, int, float, List, Optional[int]]]: Dictionary containing search results and performance metrics.
        
        - method (str): Method name
        - selected_graph_id (Optional[int]): Index of best found architecture
        - actual_y_perf_of_selected (float): Performance of selected architecture
        - selection_metric_value (float): Metric value used for selection
        - selected_graph_origin (str): Origin method name
        - discovered_count (int): Number of architectures evaluated
        - total_iterations_run (int): Number of greedy steps completed
        - rank_position_overall (float): Rank among all architectures
        - percentile_overall (float): Percentile ranking
        - total_samples_overall (int): Total available architectures
        - performance_trajectory (List): Performance over time
        - total_evaluation_time (float): Time spent on evaluations
        - total_run_time (float): Total algorithm runtime
            
    Example:
        >>> results = backward_greedy_heuristic_analysis(
        ...     dataset=dataset,
        ...     micro_action_set=micro_actions,
        ...     overall_actual_y=y_tensor,
        ...     higher_is_better=True,
        ...     termination_threshold_ratio=0.05
        ... )
        >>> print(f"Best architecture: {results['selected_graph_id']}")
        >>> print(f"Performance: {results['actual_y_perf_of_selected']:.4f}")
    """
    performance_cache = {}
    time_cache = {}
    performance_trajectory = []
    total_evaluated_count = 0
    num_total_valid_graphs = len(micro_action_set.valid_edge_sets_list)
    global_best_perf = float('-inf') if higher_is_better else float('inf')
    global_best_index = -1
    current_index = -1
    final_iteration_count = 0
    method_origin = "Backward Greedy"
    total_samples = overall_actual_y.numel() if overall_actual_y is not None else 0
    total_evaluation_time = 0.0
    
    start_time = time.time()

    termination_count_threshold = int(termination_threshold_ratio * num_total_valid_graphs)
    if termination_count_threshold < 1: termination_count_threshold = 1
    print(f"{method_name}: Budget set to {termination_count_threshold} unique evaluations.")

    full_graph_index = -1
    if hasattr(dataset, 'full_graph_id') and dataset.full_graph_id is not None:
        full_graph_index = dataset.full_graph_id

    print(f"Backward Greedy (Iter 1): Starting from Full Graph Index {full_graph_index}")
    current_index = full_graph_index

    initial_cache_size = len(performance_cache)
    initial_perf = get_performance_for_index(current_index, dataset, performance_cache)
    final_iteration_count = 1
    
    if initial_perf is not None:
        performance_cache[current_index] = initial_perf
        
        eval_time = calculate_evaluation_time(current_index, dataset, time_cache)
        if eval_time is not None:
            if len(performance_cache) > initial_cache_size:
                total_evaluation_time += eval_time

    total_evaluated_count, global_best_perf, global_best_index = \
        update_trajectory_and_best(
            current_index, initial_perf, performance_cache, initial_cache_size,
            total_evaluated_count, performance_trajectory,
            global_best_perf, global_best_index, higher_is_better
        )

    if global_best_index != current_index:
        print(f"Warning: Failed to cache performance for initial state Index {current_index}. Cannot proceed.")
        results = {
            "method": method_name, "selected_graph_id": None, "actual_y_perf_of_selected": np.nan,
            "selection_metric_value": np.nan, "selected_graph_origin": method_origin,
            "discovered_count": total_evaluated_count, "total_iterations_run": final_iteration_count,
            "rank_position_overall": np.nan, "percentile_overall": np.nan,
            "total_samples_overall": total_samples, "performance_trajectory": performance_trajectory,
            "total_evaluation_time": total_evaluation_time,
            "total_run_time": time.time() - start_time
        }
        pad_trajectory(performance_trajectory, total_evaluated_count, termination_count_threshold, method_name)
        return results
    else:
         print(f"Backward Greedy (Iter 1): Initial State Index {current_index}, Perf: {global_best_perf:.4f}. Total evaluated: {total_evaluated_count}")

    backward_actions = []
    action_names = ["remove_fk_pk_edge", "convert_row_to_edge", "convert_edge_to_row"]
    for name in action_names:
        if hasattr(micro_action_set, name):
            backward_actions.append(getattr(micro_action_set, name))

    _, final_iteration_count, performance_cache, total_evaluated_count, performance_trajectory, final_global_best_perf, final_global_best_index, time_cache, total_evaluation_time = _perform_greedy_search(
        initial_index=current_index,
        action_functions=backward_actions,
        dataset=dataset,
        micro_action_set=micro_action_set,
        performance_cache=performance_cache,
        termination_count_threshold=termination_count_threshold,
        higher_is_better=higher_is_better,
        num_total_valid_graphs=num_total_valid_graphs,
        global_best_perf=global_best_perf,
        global_best_index=global_best_index,
        total_evaluated_count=total_evaluated_count,
        performance_trajectory=performance_trajectory,
        time_cache=time_cache,
        total_evaluation_time=total_evaluation_time,
        start_iteration=final_iteration_count,
        method_name_for_print="Backward Greedy"
    )

    total_run_time = time.time() - start_time
    total_evaluation_time = sum(time_cache.get(idx, 0) for idx in performance_cache)

    final_selected_perf = performance_cache.get(final_global_best_index, np.nan) if final_global_best_index != -1 else np.nan
    results = {
        "method": method_name,
        "selected_graph_id": final_global_best_index if final_global_best_index != -1 else None,
        "actual_y_perf_of_selected": final_selected_perf,
        "selection_metric_value": final_selected_perf,
        "selected_graph_origin": method_origin,
        "discovered_count": total_evaluated_count,
        "total_iterations_run": final_iteration_count,
        "rank_position_overall": np.nan,
        "percentile_overall": np.nan,
        "total_samples_overall": total_samples,
        "performance_trajectory": performance_trajectory,
        "total_evaluation_time": total_evaluation_time,
        "total_run_time": total_run_time
    }

    pad_trajectory(performance_trajectory, total_evaluated_count, termination_count_threshold, method_name)

    if final_global_best_index != -1 and overall_actual_y is not None and not np.isnan(final_selected_perf):
        rank_info = calculate_overall_rank(
            final_selected_perf,
            overall_actual_y,
            higher_is_better
        )
        if rank_info:
            results["rank_position_overall"] = rank_info["rank_position_overall"]
            results["percentile_overall"] = rank_info["percentile_overall"]

    return results

def random_greedy_heuristic_analysis(
    dataset: PerformancePredictionDataset,
    micro_action_set: MicroActionSet,
    overall_actual_y: torch.Tensor,
    higher_is_better: bool,
    termination_threshold_ratio: float,
    method_name: str = "Local Greedy Heuristic",
):
    """
    Perform Neural Architecture Search using Random/Local Greedy Strategy.
    
    This function implements random greedy search that starts from a randomly chosen
    graph configuration and iteratively applies all types of micro actions (add/remove
    edges, conversions) to improve performance. The algorithm greedily selects the
    best local improvement at each step from the full set of available actions.
    
    Args:
        dataset (PerformancePredictionDataset): Dataset containing architecture 
            performance data
        micro_action_set (MicroActionSet): Set of micro actions for architecture
            space exploration
        overall_actual_y (torch.Tensor): Complete performance tensor for
            ranking calculations
        higher_is_better (bool): Whether higher performance values are better
        termination_threshold_ratio (float): Fraction of total architectures to
            evaluate as budget
        method_name (str): Name identifier for this method. 
            Defaults to "Local Greedy Heuristic".
            
    Returns:
        Dict[str, Union[str, int, float, List, Optional[int]]]: Dictionary containing search results and performance metrics.
        
        - method (str): Method name
        - selected_graph_id (Optional[int]): Index of best found architecture
        - actual_y_perf_of_selected (float): Performance of selected architecture
        - selection_metric_value (float): Metric value used for selection
        - selected_graph_origin (str): Origin method name
        - discovered_count (int): Number of architectures evaluated
        - total_iterations_run (int): Number of greedy steps completed
        - rank_position_overall (float): Rank among all architectures
        - percentile_overall (float): Percentile ranking
        - total_samples_overall (int): Total available architectures
        - performance_trajectory (List): Performance over time
        - total_evaluation_time (float): Time spent on evaluations
        - total_run_time (float): Total algorithm runtime
            
    Example:
        >>> results = random_greedy_heuristic_analysis(
        ...     dataset=dataset,
        ...     micro_action_set=micro_actions,
        ...     overall_actual_y=y_tensor,
        ...     higher_is_better=True,
        ...     termination_threshold_ratio=0.05
        ... )
        >>> print(f"Best architecture: {results['selected_graph_id']}")
        >>> print(f"Performance: {results['actual_y_perf_of_selected']:.4f}")
    """
    performance_cache = {}
    time_cache = {}
    performance_trajectory = []
    total_evaluated_count = 0
    num_total_valid_graphs = len(micro_action_set.valid_edge_sets_list)
    global_best_perf = float('-inf') if higher_is_better else float('inf')
    global_best_index = -1
    current_index = -1
    final_iteration_count = 0
    method_origin = "Local Greedy"
    total_samples = overall_actual_y.numel() if overall_actual_y is not None else 0
    total_evaluation_time = 0.0
    
    start_time = time.time()

    termination_count_threshold = int(termination_threshold_ratio * num_total_valid_graphs)
    if termination_count_threshold < 1: termination_count_threshold = 1
    print(f"{method_name}: Budget set to {termination_count_threshold} unique evaluations.")

    if num_total_valid_graphs > 0:
        current_index = random.randint(0, num_total_valid_graphs - 1)
        print(f"Local Greedy (Iter 1): Starting from Random Index {current_index}")
    else:
        print("Error: No valid graphs available to select a random starting point.")
        results = {
            "method": method_name, "selected_graph_id": None, "actual_y_perf_of_selected": np.nan,
            "selection_metric_value": np.nan, "selected_graph_origin": method_origin,
            "discovered_count": 0, "total_iterations_run": 0,
            "rank_position_overall": np.nan, "percentile_overall": np.nan,
            "total_samples_overall": total_samples, "performance_trajectory": []
        }
        pad_trajectory(performance_trajectory, total_evaluated_count, termination_count_threshold, method_name)
        return results

    initial_cache_size = len(performance_cache)
    initial_perf = get_performance_for_index(current_index, dataset, performance_cache)
    final_iteration_count = 1
    
    if initial_perf is not None:
        performance_cache[current_index] = initial_perf
        
        eval_time = calculate_evaluation_time(current_index, dataset, time_cache)
        if eval_time is not None:
            if len(performance_cache) > initial_cache_size:
                total_evaluation_time += eval_time

    total_evaluated_count, global_best_perf, global_best_index = \
        update_trajectory_and_best(
            current_index, initial_perf, performance_cache, initial_cache_size,
            total_evaluated_count, performance_trajectory,
            global_best_perf, global_best_index, higher_is_better
        )

    if global_best_index == -1:
         print(f"Warning: Failed to evaluate initial state Index {current_index}. Cannot proceed.")
         results = {
            "method": method_name, "selected_graph_id": None, "actual_y_perf_of_selected": np.nan,
            "selection_metric_value": np.nan, "selected_graph_origin": method_origin,
            "discovered_count": total_evaluated_count, "total_iterations_run": final_iteration_count,
             "rank_position_overall": np.nan, "percentile_overall": np.nan,
             "total_samples_overall": total_samples, "performance_trajectory": performance_trajectory
         }
         pad_trajectory(performance_trajectory, total_evaluated_count, termination_count_threshold, method_name)
         return results
    else:
        print(f"Local Greedy (Iter 1): Initial Index {current_index}, Perf: {global_best_perf:.4f}. Total evaluated: {total_evaluated_count}")

    all_actions = []
    action_names = [
        "add_fk_pk_edge",
        "remove_fk_pk_edge",
        "convert_row_to_edge", "convert_edge_to_row"
    ]
    for name in action_names:
        if hasattr(micro_action_set, name):
            all_actions.append(getattr(micro_action_set, name))

    if not all_actions:
         print("Warning: No micro-actions found in micro_action_set. Cannot perform search steps.")
         final_selected_perf = performance_cache.get(global_best_index, np.nan)
         results = {
            "method": method_name, "selected_graph_id": global_best_index,
            "actual_y_perf_of_selected": final_selected_perf,
            "selection_metric_value": final_selected_perf,
            "selected_graph_origin": method_origin,
            "discovered_count": total_evaluated_count, "total_iterations_run": final_iteration_count,
            "rank_position_overall": np.nan, "percentile_overall": np.nan,
            "total_samples_overall": total_samples, "performance_trajectory": performance_trajectory
         }
         pad_trajectory(performance_trajectory, total_evaluated_count, termination_count_threshold, method_name)
         if overall_actual_y is not None and not np.isnan(final_selected_perf):
             rank_info = calculate_overall_rank(final_selected_perf, overall_actual_y, higher_is_better)
             if rank_info:
                 results["rank_position_overall"] = rank_info["rank_position_overall"]
                 results["percentile_overall"] = rank_info["percentile_overall"]
         return results

    _, final_iteration_count, performance_cache, total_evaluated_count, performance_trajectory, final_global_best_perf, final_global_best_index, time_cache, total_evaluation_time = _perform_greedy_search(
        initial_index=current_index,
        action_functions=all_actions,
        dataset=dataset,
        micro_action_set=micro_action_set,
        performance_cache=performance_cache,
        termination_count_threshold=termination_count_threshold,
        higher_is_better=higher_is_better,
        num_total_valid_graphs=num_total_valid_graphs,
        global_best_perf=global_best_perf,
        global_best_index=global_best_index,
        total_evaluated_count=total_evaluated_count,
        performance_trajectory=performance_trajectory,
        time_cache=time_cache,
        total_evaluation_time=total_evaluation_time,
        start_iteration=final_iteration_count,
        method_name_for_print="Local Greedy"
    )

    pad_trajectory(performance_trajectory, total_evaluated_count, termination_count_threshold, method_name)

    total_run_time = time.time() - start_time

    final_selected_perf = performance_cache.get(final_global_best_index, np.nan) if final_global_best_index != -1 else np.nan
    results = {
        "method": method_name,
        "selected_graph_id": final_global_best_index if final_global_best_index != -1 else None,
        "actual_y_perf_of_selected": final_selected_perf,
        "selection_metric_value": final_selected_perf,
        "selected_graph_origin": method_origin,
        "discovered_count": total_evaluated_count,
        "total_iterations_run": final_iteration_count,
        "rank_position_overall": np.nan,
        "percentile_overall": np.nan,
        "total_samples_overall": total_samples,
        "performance_trajectory": performance_trajectory,
        "total_evaluation_time": total_evaluation_time,
        "total_run_time": total_run_time
    }

    if final_global_best_index != -1 and overall_actual_y is not None and not np.isnan(final_selected_perf):
        rank_info = calculate_overall_rank(
            final_selected_perf,
            overall_actual_y,
            higher_is_better
        )
        if rank_info:
            results["rank_position_overall"] = rank_info["rank_position_overall"]
            results["percentile_overall"] = rank_info["percentile_overall"]

    return results

def _perform_greedy_search(
    initial_index: int,
    action_functions: list,
    dataset: PerformancePredictionDataset,
    micro_action_set: MicroActionSet,
    performance_cache: dict,
    termination_count_threshold: int,
    higher_is_better: bool,
    num_total_valid_graphs: int,
    global_best_perf: float,
    global_best_index: int,
    total_evaluated_count: int,
    performance_trajectory: list,
    time_cache: dict,
    total_evaluation_time: float,
    start_iteration: int,
    method_name_for_print: str
) -> tuple[int, int, dict, int, list, float, int, dict, float]:
    """Performs the core greedy search loop, updating trajectory."""
    current_index = initial_index
    final_iteration_count = start_iteration
    current_total_evaluated_count = total_evaluated_count
    current_performance_trajectory = performance_trajectory
    current_global_best_perf = global_best_perf
    current_global_best_index = global_best_index

    print(f"{method_name_for_print}: Starting main loop from Index {current_index}. Budget: {current_total_evaluated_count}/{termination_count_threshold}.")

    while current_total_evaluated_count < termination_count_threshold:
        final_iteration_count += 1

        if not (0 <= current_index < num_total_valid_graphs):
            print(f"Error: Current index {current_index} out of bounds. Stopping.")
            break
        current_edge_set = micro_action_set.valid_edge_sets_list[current_index]

        candidate_indices = set()
        for action_fn in action_functions:
            try:
                possible_next_states = action_fn(current_edge_set)
                if possible_next_states:
                    for _, next_index in possible_next_states:
                        if 0 <= next_index < num_total_valid_graphs:
                            candidate_indices.add(next_index)
            except Exception as e:
                continue

        if not candidate_indices:
            print(f"{method_name_for_print} (Iter {final_iteration_count}): No further neighbors from index {current_index}. Stopping.")
            break

        best_neighbor_perf = float('-inf') if higher_is_better else float('inf')
        best_neighbor_index = -1
        candidate_list = list(candidate_indices)
        random.shuffle(candidate_list)

        evaluated_count_this_iter = 0
        for candidate_index in candidate_list:
            if current_total_evaluated_count >= termination_count_threshold:
                 print(f"{method_name_for_print} (Iter {final_iteration_count}): Budget reached during neighbor evaluation ({current_total_evaluated_count}/{termination_count_threshold}).")
                 break

            initial_cache_size = len(performance_cache)
            perf = get_performance_for_index(candidate_index, dataset, performance_cache)
            evaluated_count_this_iter += 1
            
            if perf is not None:
                performance_cache[candidate_index] = perf
                
                eval_time = calculate_evaluation_time(candidate_index, dataset, time_cache)
                if eval_time is not None:
                    if len(performance_cache) > initial_cache_size:
                        total_evaluation_time += eval_time

            current_total_evaluated_count, current_global_best_perf, current_global_best_index = \
                update_trajectory_and_best(
                    candidate_index, perf, performance_cache, initial_cache_size,
                    current_total_evaluated_count, current_performance_trajectory,
                    current_global_best_perf, current_global_best_index, higher_is_better
                )

            if perf is not None:
                if higher_is_better:
                    if perf > best_neighbor_perf:
                        best_neighbor_perf = perf
                        best_neighbor_index = candidate_index
                else:
                    if perf < best_neighbor_perf:
                        best_neighbor_perf = perf
                        best_neighbor_index = candidate_index

        if current_total_evaluated_count >= termination_count_threshold:
             print(f"{method_name_for_print} (Iter {final_iteration_count}): Budget reached after evaluating neighbors ({current_total_evaluated_count}/{termination_count_threshold}).")
             break

        if best_neighbor_index == -1:
            print(f"{method_name_for_print} (Iter {final_iteration_count}): No valid improving neighbor identified among {evaluated_count_this_iter} evaluated candidates from index {current_index}. Stopping.")
            break

        if current_index not in performance_cache:
             print(f"Warning: Performance for current index {current_index} not found in cache. Evaluating defensively.")
             current_perf = get_performance_for_index(current_index, dataset, performance_cache)
             
             if current_perf is not None:
                 performance_cache[current_index] = current_perf
                 
             if current_index not in performance_cache:
                 print(f"Error: Could not evaluate current index {current_index}. Cannot compare improvement. Stopping.")
                 break

        if best_neighbor_index not in performance_cache:
             print(f"Error: Best neighbor {best_neighbor_index} perf not in cache unexpectedly. Stopping.")
             break

        current_perf = performance_cache[current_index]
        neighbor_perf = performance_cache[best_neighbor_index]

        no_improvement = (higher_is_better and neighbor_perf <= current_perf) or \
                         (not higher_is_better and neighbor_perf >= current_perf)

        if no_improvement:
            print(f"{method_name_for_print} (Iter {final_iteration_count}): Best evaluated neighbor (Idx {best_neighbor_index}, Perf: {neighbor_perf:.4f}) doesn't improve on current (Idx {current_index}, Perf: {current_perf:.4f}). Stopping.")
            break

        print(f"{method_name_for_print} (Iter {final_iteration_count}): Moving from Index {current_index} (Perf: {current_perf:.4f}) to {best_neighbor_index} (Perf: {neighbor_perf:.4f}). Budget: {current_total_evaluated_count}/{termination_count_threshold}")
        current_index = best_neighbor_index

    print(f"\n{method_name_for_print} Search loop finished after {final_iteration_count} iterations.")
    print(f"Total unique graphs evaluated (budget used): {current_total_evaluated_count}")
    if current_global_best_index != -1:
         final_best_perf_display = performance_cache.get(current_global_best_index, current_global_best_perf)
         print(f"Final Global Best: Index {current_global_best_index}, Performance: {final_best_perf_display:.4f}")
    else:
         print("Final Global Best: None found or evaluated.")

    return current_index, final_iteration_count, performance_cache, current_total_evaluated_count, current_performance_trajectory, current_global_best_perf, current_global_best_index, time_cache, total_evaluation_time
