import torch
import random
import numpy as np
import time

from ..dataset import PerformancePredictionDataset
from ..micro_action import MicroActionSet
from .utils import calculate_overall_rank, get_performance_for_index, update_trajectory_and_best, pad_trajectory, calculate_evaluation_time

def random_heuristic_analysis(
    dataset: PerformancePredictionDataset,
    micro_action_set: MicroActionSet, 
    overall_actual_y: torch.Tensor,
    higher_is_better: bool,
    termination_threshold_ratio: float,
    method_name: str = "Random Heuristic",
):
    """Performs sequential evaluation of graphs chosen randomly from the entire valid graph pool"""
    performance_cache = {}
    time_cache = {}
    performance_trajectory = []
    total_evaluated_count = 0
    best_perf_so_far = float('-inf') if higher_is_better else float('inf')
    best_index_so_far = -1
    method_origin = "Random Heuristic"
    total_samples_overall = overall_actual_y.numel() if overall_actual_y is not None else 0
    total_evaluation_time = 0.0

    start_time = time.time()

    num_total_valid_graphs = len(micro_action_set.valid_edge_sets_list)
    if num_total_valid_graphs == 0:
        print(f"Error: No valid graphs available in MicroActionSet. Cannot proceed.")
        return {
            "method": method_name, "selected_graph_id": None, "actual_y_perf_of_selected": np.nan,
            "selection_metric_value": np.nan, "selected_graph_origin": method_origin,
            "discovered_count": 0, "total_iterations_run": 0,
            "rank_position_overall": np.nan, "percentile_overall": np.nan,
            "total_samples_overall": total_samples_overall, "performance_trajectory": [],
            "total_evaluation_time": 0.0, "total_run_time": 0.0
        }

    evaluation_budget = max(1, int(termination_threshold_ratio * num_total_valid_graphs))
    print(f"{method_name}: Total valid graphs: {num_total_valid_graphs}. Budget set to {evaluation_budget} unique evaluations.")

    all_valid_indices = list(range(num_total_valid_graphs))
    shuffled_indices = random.sample(all_valid_indices, k=num_total_valid_graphs)

    for index in shuffled_indices:
        if total_evaluated_count >= evaluation_budget:
            print(f"{method_name}: Budget reached ({total_evaluated_count}/{evaluation_budget}).")
            break

        initial_cache_size = len(performance_cache)
        perf = get_performance_for_index(index, dataset, performance_cache)

        if perf is not None:
            performance_cache[index] = perf
            
            eval_time = calculate_evaluation_time(index, dataset, time_cache)
            if eval_time is not None:
                if len(performance_cache) > initial_cache_size:
                    total_evaluation_time += eval_time
            
        total_evaluated_count, best_perf_so_far, best_index_so_far = \
            update_trajectory_and_best(
                index, perf, performance_cache, initial_cache_size,
                total_evaluated_count, performance_trajectory,
                best_perf_so_far, best_index_so_far, higher_is_better
            )

    print(f"{method_name}: Finished evaluating {total_evaluated_count} graphs.")

    total_run_time = time.time() - start_time

    pad_trajectory(performance_trajectory, total_evaluated_count, evaluation_budget, method_name)

    final_selected_perf = best_perf_so_far if best_index_so_far != -1 else np.nan
    final_selected_index = best_index_so_far if best_index_so_far != -1 else None

    results = {
        "method": method_name,
        "selected_graph_id": final_selected_index,
        "actual_y_perf_of_selected": final_selected_perf,
        "selection_metric_value": final_selected_perf,
        "selected_graph_origin": method_origin,
        "discovered_count": total_evaluated_count,
        "total_iterations_run": total_evaluated_count,
        "rank_position_overall": np.nan,
        "percentile_overall": np.nan,
        "total_samples_overall": total_samples_overall,
        "performance_trajectory": performance_trajectory,
        "total_evaluation_time": total_evaluation_time,
        "total_run_time": total_run_time
    }

    if final_selected_index is not None and not np.isnan(final_selected_perf) and overall_actual_y is not None:
        rank_info = calculate_overall_rank(
            final_selected_perf,
            overall_actual_y,
            higher_is_better
        )
        if rank_info:
            results["rank_position_overall"] = rank_info["rank_position_overall"]
            results["percentile_overall"] = rank_info["percentile_overall"]

    return results