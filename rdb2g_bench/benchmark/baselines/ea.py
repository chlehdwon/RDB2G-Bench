import torch
import numpy as np
import random
import time
from typing import Dict, Union, List, Optional

from ..dataset import PerformancePredictionDataset
from ..micro_action import MicroActionSet
from .utils import calculate_overall_rank, get_performance_for_index, update_trajectory_and_best, pad_trajectory, calculate_evaluation_time

def evolutionary_heuristic_analysis(
    dataset: PerformancePredictionDataset,
    micro_action_set: MicroActionSet,
    overall_actual_y: torch.Tensor,
    higher_is_better: bool,
    termination_threshold_ratio: float,
    method_name: str = "Evolutionary Heuristic",
    population_size: int = 10,
    tournament_size: int = 10,
    max_iterations: int = 1000,
):
    """
    Perform Neural Architecture Search using Evolutionary Algorithm.
    
    This function implements a complete evolutionary algorithm for finding optimal
    graph neural network architectures. It maintains a population of architectures,
    applies micro action-based mutations, and uses tournament selection for evolution.
    
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
            Defaults to "Evolutionary Heuristic".
        population_size (int): Number of individuals in the population.
            Defaults to 10.
        tournament_size (int): Number of individuals selected for tournament.
            Defaults to 10.
        max_iterations (int): Maximum number of evolutionary generations.
            Defaults to 1000.
            
    Returns:
        Dict[str, Union[str, int, float, List, Optional[int]]]: Dictionary containing search results and performance metrics.
        
        - method (str): Method name
        - selected_graph_id (Optional[int]): Index of best found architecture
        - actual_y_perf_of_selected (float): Performance of selected architecture
        - selection_metric_value (float): Metric value used for selection
        - selected_graph_origin (str): Origin method name
        - discovered_count (int): Number of architectures evaluated
        - total_iterations_run (int): Number of generations completed
        - rank_position_overall (float): Rank among all architectures
        - percentile_overall (float): Percentile ranking
        - total_samples_overall (int): Total available architectures
        - performance_trajectory (List): Performance over time
        - total_evaluation_time (float): Time spent on evaluations
        - total_run_time (float): Total algorithm runtime
            
    Example:
        >>> results = evolutionary_heuristic_analysis(
        ...     dataset=dataset,
        ...     micro_action_set=micro_actions,
        ...     overall_actual_y=y_tensor,
        ...     higher_is_better=True,
        ...     termination_threshold_ratio=0.05,
        ...     population_size=20,
        ...     tournament_size=5,
        ...     max_iterations=100
        ... )
        >>> print(f"Best architecture: {results['selected_graph_id']}")
        >>> print(f"Performance: {results['actual_y_perf_of_selected']:.4f}")
    """
    population = []
    seen_indices = set()
    performance_cache = {}
    time_cache = {}
    performance_trajectory = []
    num_total_valid_graphs = len(micro_action_set.valid_edge_sets_list)
    discovered_count = 0
    best_perf_so_far = float('-inf') if higher_is_better else float('inf')
    best_index_so_far = -1
    final_iteration_count = 0
    total_evaluated_count = 0
    total_evaluation_time = 0.0
    
    start_time = time.time()

    if num_total_valid_graphs == 0:
        print("Error: No valid graphs found in MicroActionSet. Cannot initialize population.")
        initial_indices = []
    else:
        pop_size_to_init = min(population_size, num_total_valid_graphs)
        if pop_size_to_init <= 0:
             print("Warning: Cannot initialize population with size <= 0.")
             initial_indices = []
        elif pop_size_to_init > num_total_valid_graphs:
             print(f"Warning: Requested population size ({population_size}) exceeds number of valid graphs ({num_total_valid_graphs}). Using {num_total_valid_graphs}.")
             pop_size_to_init = num_total_valid_graphs
             initial_indices = np.random.choice(
                 num_total_valid_graphs,
                 size=pop_size_to_init,
                 replace=False
             )
        else:
            initial_indices = np.random.choice(
                num_total_valid_graphs,
                size=pop_size_to_init,
                replace=False
            )

    print(f"Initializing population with {len(initial_indices)} individuals...")
    for index in initial_indices:
        population.append((index, 0))
        if index not in seen_indices:
            seen_indices.add(index)
            discovered_count += 1

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

    print(f"After initialization ({total_evaluated_count} evals): Best perf={best_perf_so_far:.4f}")

    termination_count_threshold = int(termination_threshold_ratio * num_total_valid_graphs)
    print(f"Evolution budget set to {termination_count_threshold} unique evaluations.")

    for iteration in range(max_iterations):
        final_iteration_count = iteration + 1
        if not population:
            break

        mutated_indices_this_step = []

        current_pop_size = len(population)
        actual_tournament_size = min(tournament_size, current_pop_size)
        if actual_tournament_size <= 0:
            break

        tournament_candidate_indices = np.random.choice(
            current_pop_size,
            size=actual_tournament_size,
            replace=False
        )

        for pop_list_idx in tournament_candidate_indices:
            current_index, _ = population[pop_list_idx]
            current_edge_set = micro_action_set.valid_edge_sets_list[current_index]

            action_fns = [
                micro_action_set.add_fk_pk_edge,
                micro_action_set.remove_fk_pk_edge,
                micro_action_set.convert_row_to_edge,
                micro_action_set.convert_edge_to_row,
            ]
            random.shuffle(action_fns)

            mutated_index = current_index
            for chosen_action_fn in action_fns:
                possible_next_states = chosen_action_fn(current_edge_set)

                if possible_next_states:
                    _, next_index = random.choice(possible_next_states)
                    mutated_index = next_index

                    if mutated_index not in seen_indices:
                        seen_indices.add(mutated_index)
                        discovered_count += 1
                    break

            mutated_indices_this_step.append(mutated_index)

        tournament_performances = []
        tournament_indices = mutated_indices_this_step
        valid_tournament_results = []

        terminated_early = False
        for index in tournament_indices:
            initial_cache_size = len(performance_cache)
            perf = get_performance_for_index(index, dataset, performance_cache)

            if perf is not None:
                tournament_performances.append(perf)
                valid_tournament_results.append((perf, index))
                
                performance_cache[index] = perf

                eval_time = calculate_evaluation_time(index, dataset, time_cache)
                if eval_time is not None:
                    if len(performance_cache) > initial_cache_size:
                        total_evaluation_time += eval_time

                new_total_evaluated_count, new_best_perf_so_far, new_best_index_so_far = \
                    update_trajectory_and_best(
                        index, perf, performance_cache, initial_cache_size,
                        total_evaluated_count, performance_trajectory,
                        best_perf_so_far, best_index_so_far, higher_is_better
                    )

                if len(performance_cache) > initial_cache_size:
                    total_evaluated_count = new_total_evaluated_count
                    best_perf_so_far = new_best_perf_so_far
                    best_index_so_far = new_best_index_so_far

                    if total_evaluated_count >= termination_count_threshold:
                        print(f"Termination condition met at iteration {iteration+1}: Evaluated {total_evaluated_count} unique graphs >= threshold {termination_count_threshold}")
                        terminated_early = True
                        break
                else:
                    best_perf_so_far = new_best_perf_so_far
                    best_index_so_far = new_best_index_so_far

        if terminated_early:
            break

        if not valid_tournament_results:
            print(f"Warning: Iteration {iteration+1} - No valid performances in tournament. Skipping replacement.")
            continue

        if higher_is_better:
            best_tournament_perf, best_tournament_index = max(valid_tournament_results, key=lambda item: item[0])
        else:
            best_tournament_perf, best_tournament_index = min(valid_tournament_results, key=lambda item: item[0])

        oldest_pop_list_idx = -1
        max_age = -1
        for i, (_, age) in enumerate(population):
            if age > max_age:
                max_age = age
                oldest_pop_list_idx = i

        if oldest_pop_list_idx != -1:
            population[oldest_pop_list_idx] = (best_tournament_index, 0)

        for i in range(len(population)):
             if i != oldest_pop_list_idx:
                 idx, age = population[i]
                 population[i] = (idx, age + 1)

    final_selected_index = best_index_so_far
    final_selected_perf = best_perf_so_far

    pad_trajectory(performance_trajectory, total_evaluated_count, termination_count_threshold, method_name)

    total_run_time = time.time() - start_time

    results = {
        "method": method_name,
        "selected_graph_id": final_selected_index,
        "actual_y_perf_of_selected": final_selected_perf,
        "selection_metric_value": final_selected_perf,
        "selected_graph_origin": "Evolutionary",
        "discovered_count": discovered_count,
        "total_iterations_run": final_iteration_count,
        "rank_position_overall": np.nan,
        "percentile_overall": np.nan,
        "total_samples_overall": overall_actual_y.numel() if overall_actual_y is not None else 0,
        "performance_trajectory": performance_trajectory,
        "total_evaluation_time": total_evaluation_time,
        "total_run_time": total_run_time
    }

    rank_info = calculate_overall_rank(
        final_selected_perf,
        overall_actual_y,
        higher_is_better
    )
    if rank_info:
        results["rank_position_overall"] = rank_info["rank_position_overall"]
        results["percentile_overall"] = rank_info["percentile_overall"]

    return results
