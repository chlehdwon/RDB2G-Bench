import argparse
import json
import os
# os.environ['XDG_CACHE_HOME'] = '/data/cache'

from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig

from pathlib import Path
import numpy as np
import torch
from torch_geometric.seed import seed_everything

from .dataset import PerformancePredictionDataset
from .micro_action import MicroActionSet

from ..common.text_embedder import GloveTextEmbedding
from ..common.search_space.gnn_search_space import GNNNodeSearchSpace, IDGNNLinkSearchSpace

from relbench.base import TaskType
from relbench.tasks import get_task
from relbench.datasets import get_dataset

from .baselines.ea import evolutionary_heuristic_analysis
from .baselines.random import random_heuristic_analysis
from .baselines.greedy import forward_greedy_heuristic_analysis, backward_greedy_heuristic_analysis, random_greedy_heuristic_analysis
from .baselines.rl import rl_heuristic_analysis
from .baselines.bo import bayesian_optimization_analysis
from .baselines.utils import calculate_overall_rank

def full_graph_heuristic_analysis(
    dataset: PerformancePredictionDataset,
    overall_actual_y: torch.Tensor,
    higher_is_better: bool,
    method_name: str = "Full Graph"
):
    """
    Analyze the performance of using the full graph configuration as a heuristic baseline.
    
    This function evaluates the performance of the predefined full graph configuration,
    which includes all possible edges in the search space. It serves as an important
    baseline for comparison with other heuristic search methods.
    
    Args:
        dataset (PerformancePredictionDataset): The performance prediction dataset containing
            experimental results and full graph configuration information.
        overall_actual_y (torch.Tensor): Tensor containing actual performance values for
            all graph configurations in the dataset.
        higher_is_better (bool): Whether higher performance values indicate better results.
            True for metrics like accuracy/ROC-AUC, False for metrics like MAE/loss.
        method_name (str, optional): Name identifier for this analysis method.
            Defaults to "Full Graph".
    
    Returns:
        dict or None: Dictionary containing analysis results, or None if full graph
        configuration is not available or accessible.
        
        The returned dictionary contains:
            - method (str): Method name identifier
            - selected_graph_id (int): ID of the full graph configuration
            - actual_y_perf_of_selected (float): Actual performance of full graph
            - selected_graph_origin (str): Source description ("Predefined Full Graph")
            - selection_metric_value (None): Not applicable for predefined configuration
            - rank_position_overall (int): Rank position among all configurations
            - percentile_overall (float): Percentile ranking (0-100)
            - total_samples_overall (int): Total number of configurations evaluated
    
    Example:
        >>> import torch
        >>> from rdb2g_bench.benchmark.dataset import PerformancePredictionDataset
        >>> 
        >>> # Load dataset and performance data
        >>> dataset = PerformancePredictionDataset("rel-f1", "driver-top3")
        >>> performance_values = torch.tensor([0.85, 0.92, 0.78, 0.89])
        >>> 
        >>> # Analyze full graph performance
        >>> results = full_graph_heuristic_analysis(
        ...     dataset=dataset,
        ...     overall_actual_y=performance_values,
        ...     higher_is_better=True
        ... )
        >>> 
        >>> if results:
        ...     print(f"Full graph rank: {results['rank_position_overall']}")
        ...     print(f"Performance: {results['actual_y_perf_of_selected']:.4f}")
    """
    if not hasattr(dataset, 'full_graph_id') or dataset.full_graph_id is None:
        print("Warning: Full graph ID not set in dataset. Skipping full graph heuristic.")
        return None
    full_graph_id = dataset.full_graph_id

    try:
        full_graph_row = dataset.df_result_group[dataset.df_result_group['idx'] == full_graph_id]
        if full_graph_row.empty:
            print(f"Error: Full graph ID {full_graph_id} not found in df_result_group. Skipping.")
            return None
        if dataset.target_col not in full_graph_row.columns:
            print(f"Error: Target column '{dataset.target_col}' not found. Skipping.")
            return None
        full_graph_actual_y_perf = full_graph_row.iloc[0][dataset.target_col]
    except Exception as e:
        print(f"Error accessing performance for full_graph_id {full_graph_id}: {e}. Skipping.")
        return None

    results = {"method": method_name}
    results["selected_graph_id"] = full_graph_id
    results["actual_y_perf_of_selected"] = full_graph_actual_y_perf
    results["selected_graph_origin"] = "Predefined Full Graph"
    results["selection_metric_value"] = None

    rank_info = calculate_overall_rank(
        full_graph_actual_y_perf,
        overall_actual_y,
        higher_is_better
    )
    if rank_info: results.update(rank_info)

    return results

def main(args):
    """
    Execute the complete RDB2G-Bench benchmarking pipeline.
    
    This function orchestrates the entire benchmarking process, including dataset preparation,
    search space initialization, execution of multiple heuristic algorithms, and comprehensive
    performance analysis. It supports multiple independent runs for statistical significance
    and provides detailed trajectory tracking and result aggregation.
    
    Args:
        args: Parsed command-line arguments containing benchmark configuration.
        
        Expected attributes in args:
            - dataset (str): Dataset name (e.g., "rel-f1", "rel-avito")
            - task (str): Task name (e.g., "driver-top3", "user-ad-visit")  
            - method (list): List of methods to execute ("all", "ea", "greedy", "rl", "bo")
            - num_runs (int): Number of independent runs for statistical analysis
            - budget_percentage (float): Budget as fraction of total search space (0.0-1.0)
            - seed (int): Base random seed for reproducibility
            - cache_dir (str): Directory for caching processed data
            - result_dir (str): Directory for saving benchmark results
            - tag (str): Experiment tag for organizing results
    
    Returns:
        None: Results are saved to CSV files and printed to console.
        
        Generated output files:
            - Individual trajectory CSVs for each method
            - Combined trajectories CSV for all methods
            - Performance summary CSV with final results
            - Console output with detailed analysis

    Example:
        >>> import argparse
        >>> from rdb2g_bench.benchmark.benchmark import main
        >>> 
        >>> # Create argument parser and set configuration
        >>> parser = argparse.ArgumentParser()
        >>> # ... add argument definitions ...
        >>> args = parser.parse_args([
        ...     '--dataset', 'rel-f1',
        ...     '--task', 'driver-top3', 
        ...     '--method', 'ea', 'greedy',
        ...     '--num_runs', '3',
        ...     '--budget_percentage', '0.1'
        ... ])
        >>> 
        >>> # Execute benchmark
        >>> main(args)
        >>> # Results saved to CSV files and printed to console
    
    Output Structure:
        The function generates comprehensive output including:
        
        **Output Logs**:
            - Progress updates for each run and method
            - Performance summaries with rankings and percentiles
            - Execution time statistics
            - Method comparison analysis
        
        **CSV Files**:
            - `avg_trajectory_{method}_{num_runs}runs.csv`: Average performance trajectories
            - `all_methods_trajectories_{num_runs}runs.csv`: Combined trajectory data
            - `performance_summary_{num_runs}runs.csv`: Final performance comparison
    """
    all_runs_results = {}
    original_overall_actual_y = None
    original_overall_graph_ids = None
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(args.dataset, download=True)
    task = get_task(args.dataset, args.task, download=True)
    task_type = task.task_type
    stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(dataset.get_db())
        Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)

    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.get_db(),
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device), batch_size=256
        ),
        cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
    )

    if task_type == TaskType.BINARY_CLASSIFICATION:
        higher_is_better = True
        gnn_space_class = GNNNodeSearchSpace
        src_table = task.entity_table
        dst_table = None
    elif task_type == TaskType.REGRESSION:
        higher_is_better = False
        gnn_space_class = GNNNodeSearchSpace
        src_table = task.entity_table
        dst_table = None
    elif task_type == TaskType.MULTILABEL_CLASSIFICATION:
        higher_is_better = True
        gnn_space_class = GNNNodeSearchSpace
        src_table = task.entity_table
        dst_table = None
    elif task_type == TaskType.LINK_PREDICTION:
        higher_is_better = True
        gnn_space_class = IDGNNLinkSearchSpace
        src_table = task.src_entity_table
        dst_table = task.dst_entity_table
        if src_table is None or dst_table is None:
             raise ValueError("Link prediction task missing source_table or destination_table attribute.")
    else:
        raise ValueError(f"Task type {task_type} is unsupported for determining GNNSpaceClass and tables.")

    print("Initializing MicroActionSet...")
    micro_action_set = MicroActionSet(
        dataset=args.dataset,
        task=args.task,
        hetero_data=data,
        GNNSpaceClass=gnn_space_class,
        num_layers=2,
        src_entity_table=src_table,
        dst_entity_table=dst_table
    )
    print("MicroActionSet initialized.")

    print("Initializing dataset...")
    base_seed = args.seed
    perf_pred_dataset = PerformancePredictionDataset(
        dataset_name=args.dataset,
        task_name=args.task,
        tag=args.tag,
        cache_dir=args.cache_dir,
        result_dir=args.result_dir,
        seed=base_seed,
        device=str(device)
    )
    print("Dataset initialized.")
    
    df_results = perf_pred_dataset.df_result_group
    target_col = perf_pred_dataset.target_col
    id_col = 'idx'

    for run_index in range(args.num_runs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        current_seed = args.seed + run_index
        print(f"\n--- Starting Run {run_index + 1}/{args.num_runs} (Seed: {current_seed}) ---")
        
        seed_everything(current_seed)
        
        overall_actual_y_run = torch.tensor(df_results[target_col].values.copy(), dtype=torch.float).cpu()
        overall_graph_ids_run = torch.tensor(df_results[id_col].values.copy(), dtype=torch.long).cpu()
        
        print(f"Extracted performance data: {overall_actual_y_run.numel()} samples")

        if run_index == 0:
            original_overall_actual_y = overall_actual_y_run
            original_overall_graph_ids = overall_graph_ids_run
            print(f"Stored original dataset distribution from run 0 ({original_overall_actual_y.numel()} samples) for final ranking.")

        print("\nAnalyzing results for this run...")
        run_analysis_results = {}

        print("Calculating Actual Best (Overall Dataset)...")
        if overall_actual_y_run.numel() > 0:
            if higher_is_better:
                best_idx_run = torch.argmax(overall_actual_y_run).item()
            else:
                best_idx_run = torch.argmin(overall_actual_y_run).item()

            best_perf_run = overall_actual_y_run[best_idx_run].item()
            best_id_run = overall_graph_ids_run[best_idx_run].item()
            actual_best_rank_info = calculate_overall_rank(
                best_perf_run, overall_actual_y_run, higher_is_better
            )
            run_analysis_results["Actual Best (Overall Dataset)"] = {
                "method": "Actual Best (Overall Dataset)",
                "selected_graph_id": best_id_run,
                "actual_y_perf_of_selected": best_perf_run,
                "selection_metric_value": None,
                "selected_graph_origin": "Overall Dataset (Actual)",
                "rank_position_overall": actual_best_rank_info["rank_position_overall"],
                "percentile_overall": actual_best_rank_info["percentile_overall"], 
                "total_samples_overall": actual_best_rank_info["total_samples_overall"],
            }
        print("Actual Best calculation finished.")

        print("Running Random Heuristic analysis...")
        res_random = random_heuristic_analysis(
            dataset=perf_pred_dataset,
            micro_action_set=micro_action_set,
            overall_actual_y=overall_actual_y_run,
            higher_is_better=higher_is_better,
            termination_threshold_ratio=args.budget_percentage,
        )
        if res_random: run_analysis_results[res_random["method"]] = res_random
        print("Random Heuristic analysis finished.")
        
        print("Running Random Sequential analysis...")
        res_random_seq = random_heuristic_analysis(
            dataset=perf_pred_dataset,
            micro_action_set=micro_action_set,
            overall_actual_y=overall_actual_y_run,
            higher_is_better=higher_is_better,
            termination_threshold_ratio=args.budget_percentage,
        )
        if res_random_seq: run_analysis_results[res_random_seq["method"]] = res_random_seq
        print("Random Sequential analysis finished.")

        print("Running Full Graph Heuristic analysis...")
        res_full = full_graph_heuristic_analysis(
            perf_pred_dataset, 
            overall_actual_y=overall_actual_y_run,
            higher_is_better=higher_is_better
        )
        if res_full: run_analysis_results[res_full["method"]] = res_full
        print("Full Graph Heuristic analysis finished.")

        if 'all' in args.method or 'ea' in args.method:
            print("Running Evolutionary Heuristic analysis...")
            res_evo = evolutionary_heuristic_analysis(
                dataset=perf_pred_dataset,
                micro_action_set=micro_action_set, 
                overall_actual_y=overall_actual_y_run,
                higher_is_better=higher_is_better,
                termination_threshold_ratio=args.budget_percentage,
            )
            if res_evo: run_analysis_results[res_evo["method"]] = res_evo
            print("Evolutionary Heuristic analysis finished.")

        if 'all' in args.method or 'greedy' in args.method:
            print("Running Greedy Heuristic analysis...")
            res_greedy_forward = forward_greedy_heuristic_analysis(
                dataset=perf_pred_dataset,
                micro_action_set=micro_action_set,
                overall_actual_y=overall_actual_y_run,
                higher_is_better=higher_is_better,
                termination_threshold_ratio=args.budget_percentage,
            )
            if res_greedy_forward: run_analysis_results[res_greedy_forward["method"]] = res_greedy_forward
            res_greedy_backward = backward_greedy_heuristic_analysis(
                dataset=perf_pred_dataset,
                micro_action_set=micro_action_set,
                overall_actual_y=overall_actual_y_run,
                higher_is_better=higher_is_better,
                termination_threshold_ratio=args.budget_percentage,
            ) 
            if res_greedy_backward: run_analysis_results[res_greedy_backward["method"]] = res_greedy_backward
            print("Greedy Heuristic analysis finished.")
            res_random_greedy = random_greedy_heuristic_analysis(
                dataset=perf_pred_dataset,
                micro_action_set=micro_action_set,
                overall_actual_y=overall_actual_y_run,
                higher_is_better=higher_is_better,
                termination_threshold_ratio=args.budget_percentage,
            )
            if res_random_greedy: run_analysis_results[res_random_greedy["method"]] = res_random_greedy
            print("Random Greedy Heuristic analysis finished.")

        if 'all' in args.method or 'rl' in args.method:
            print("Running RL Heuristic analysis...")
            res_rl = rl_heuristic_analysis(
                dataset=perf_pred_dataset,
                micro_action_set=micro_action_set,
                overall_actual_y=overall_actual_y_run,
                higher_is_better=higher_is_better,
                termination_threshold_ratio=args.budget_percentage,
            )
            if res_rl: run_analysis_results[res_rl["method"]] = res_rl
            print("RL Heuristic analysis finished.")

        if 'all' in args.method or 'bo' in args.method:
            print("Running Bayesian Optimization Heuristic analysis...")
            res_bo = bayesian_optimization_analysis(
                dataset=perf_pred_dataset,
                micro_action_set=micro_action_set,
                overall_actual_y=overall_actual_y_run,
                higher_is_better=higher_is_better,
                termination_threshold_ratio=args.budget_percentage,
            )
            if res_bo: run_analysis_results[res_bo["method"]] = res_bo
            print("Bayesian Optimization Heuristic analysis finished.")

        for method, result_dict in run_analysis_results.items():
            if method not in all_runs_results:
                all_runs_results[method] = []
            all_runs_results[method].append(result_dict)

        print(f"--- Finished Run {run_index + 1}/{args.num_runs} ---")
        

    executed_methods = list(all_runs_results.keys())
    print(f"\n--- Aggregated Performance Analysis (Methods: {executed_methods}, Avg over {args.num_runs} runs) ---")

    if original_overall_actual_y is None:
        print("Warning: Original dataset distribution not captured. Cannot rank average performance against it.")

    aggregated_results_list = []

    for method, run_results in all_runs_results.items():
        if not run_results: continue

        avg_perf = np.mean([r.get("actual_y_perf_of_selected", np.nan) for r in run_results])

        if original_overall_actual_y is not None and not np.isnan(avg_perf):
            rank_info_avg = calculate_overall_rank(
                avg_perf,
                original_overall_actual_y,
                higher_is_better
            )
        else:
            rank_info_avg = None

        avg_rank_pos = rank_info_avg["rank_position_overall"] if rank_info_avg else np.nan
        avg_percentile = rank_info_avg["percentile_overall"] if rank_info_avg else np.nan
        total_samples = rank_info_avg["total_samples_overall"] if rank_info_avg else (original_overall_actual_y.numel() if original_overall_actual_y is not None else 0)

        selected_ids = [r.get("selected_graph_id", "N/A") for r in run_results]
        selection_metrics = [r.get("selection_metric_value") for r in run_results]
        origins = list(set(r.get("selected_graph_origin", "N/A") for r in run_results))
        valid_selection_metrics = [m for m in selection_metrics if m is not None]
        avg_selection_metric = np.mean(valid_selection_metrics) if valid_selection_metrics else None
        
        eval_times = [r.get("total_evaluation_time", 0.0) for r in run_results]
        run_times = [r.get("total_run_time", 0.0) for r in run_results]
        avg_eval_time = np.mean(eval_times) if eval_times else 0.0
        avg_run_time = np.mean(run_times) if run_times else 0.0

        aggregated_results_list.append({
            "method": method,
            "avg_actual_y_perf_of_selected": avg_perf,
            "avg_rank_position_overall": avg_rank_pos,
            "avg_percentile_overall": avg_percentile,
            "total_samples_overall": total_samples, 
            "selected_graph_ids_runs": selected_ids,
            "avg_selection_metric_value": avg_selection_metric,
            "selected_graph_origins": origins,
            "avg_evaluation_time": avg_eval_time,
            "avg_run_time": avg_run_time
        })

    aggregated_results_list.sort(key=lambda x: x.get('avg_rank_position_overall', float('inf')))

    for agg_results in aggregated_results_list:
        method = agg_results['method']
        print(f"\n[{method}] (Avg over {args.num_runs} runs)")

        avg_perf = agg_results['avg_actual_y_perf_of_selected']
        avg_rank_pos = agg_results['avg_rank_position_overall']
        avg_percentile = agg_results['avg_percentile_overall']
        total_samples = agg_results['total_samples_overall']
        selected_ids = agg_results['selected_graph_ids_runs']
        avg_selection_metric = agg_results['avg_selection_metric_value']
        origins = agg_results['selected_graph_origins']
        avg_eval_time = agg_results['avg_evaluation_time']
        avg_run_time = agg_results['avg_run_time']

        print(f"  Selected Graph IDs (across runs): {selected_ids}") 
        print(f"  Origin(s): {origins}")

        if avg_selection_metric is not None:
             print(f"  Avg. Selection Metric Value: {avg_selection_metric:.4f}")

        if not np.isnan(avg_perf):
            print(f"  Avg. Actual Y Performance: {avg_perf:.4f}")
        else:
            print(f"  Avg. Actual Y Performance: N/A")

        if not np.isnan(avg_rank_pos) and not np.isnan(avg_percentile) and total_samples > 0:
            print(f"    -> Rank of Avg. Perf. (among {total_samples} original samples): Top {avg_percentile:.2f}% (Rank {round(avg_rank_pos)})")
        else:
            print("    -> Rank of Avg. Perf. vs Original Samples: N/A")
            
        print(f"  Avg. Execution Time:")
        print(f"    -> Evaluation Time: {avg_eval_time:.2f} seconds")
        print(f"    -> Total Run Time: {avg_run_time:.2f} seconds")

        print(f"\n--- Saving Average Trajectory Data to CSV ---")
        try:
            import pandas as pd
            
            csv_dir = os.path.join(args.result_dir if args.result_dir else '.', f'benchmark/{args.dataset}/{args.task}/{args.tag}')
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            
            for method, run_results in all_runs_results.items():
                if not run_results or 'performance_trajectory' not in run_results[0]:
                    continue
                
                all_trajectories_perf = []
                expected_length = None
                for run_idx, result_dict in enumerate(run_results):
                    trajectory = result_dict.get('performance_trajectory')
                    if trajectory:
                        if expected_length is None:
                            expected_length = len(trajectory)
                            if expected_length == 0:
                                expected_length = None
                                break
                        
                        if len(trajectory) == expected_length:
                            perfs = []
                            for step_data in trajectory:
                                if len(step_data) >= 2 and isinstance(step_data[1], (int, float)) and np.isfinite(step_data[1]):
                                    perfs.append(float(step_data[1]))
                                else:
                                    perfs.append(np.nan)
                            all_trajectories_perf.append(perfs)
                
                if not all_trajectories_perf or expected_length is None:
                    continue
                
                trajectories_array = np.array(all_trajectories_perf)
                avg_trajectory_perf = np.nanmean(trajectories_array, axis=0)
                std_trajectory_perf = np.nanstd(trajectories_array, axis=0)
                
                max_percent = args.budget_percentage * 100
                budget_steps = np.arange(1, expected_length + 1)
                budget_percents = (budget_steps / expected_length) * max_percent
                
                df = pd.DataFrame({
                    'Budget_Step': budget_steps,
                    'Budget_Percent': budget_percents,
                    'Avg_Performance': avg_trajectory_perf,
                    'Std_Performance': std_trajectory_perf,
                    'Num_Runs': len(all_trajectories_perf)
                })
                
                csv_filename = f'avg_trajectory_{method.replace(" ", "_")}_{args.num_runs}runs.csv'
                csv_path = os.path.join(csv_dir, csv_filename)
                df.to_csv(csv_path, index=False)
                print(f"Saved average trajectory for {method} to: {csv_path}")
            
            combined_data = []
            
            for method, run_results in all_runs_results.items():
                if not run_results or 'performance_trajectory' not in run_results[0]:
                    continue
                
                all_trajectories_perf = []
                expected_length = None
                
                for run_idx, result_dict in enumerate(run_results):
                    trajectory = result_dict.get('performance_trajectory')
                    if trajectory and (expected_length is None or len(trajectory) == expected_length):
                        if expected_length is None:
                            expected_length = len(trajectory)
                            if expected_length == 0:
                                expected_length = None
                                break
                        
                        perfs = []
                        for step_data in trajectory:
                            if len(step_data) >= 2 and isinstance(step_data[1], (int, float)) and np.isfinite(step_data[1]):
                                perfs.append(float(step_data[1]))
                            else:
                                perfs.append(np.nan)
                        all_trajectories_perf.append(perfs)
                
                if not all_trajectories_perf or expected_length is None:
                    continue
                
                trajectories_array = np.array(all_trajectories_perf)
                avg_trajectory_perf = np.nanmean(trajectories_array, axis=0)
                
                for step in range(expected_length):
                    budget_percent = (step + 1) / expected_length * max_percent
                    combined_data.append({
                        'Method': method,
                        'Budget_Step': step + 1,
                        'Budget_Percent': budget_percent,
                        'Avg_Performance': avg_trajectory_perf[step]
                    })
            
            for fixed_method in ['Actual Best (Overall Dataset)', 'Full Graph']:
                agg_result = next((item for item in aggregated_results_list if item['method'] == fixed_method), None)
                if agg_result:
                    avg_perf = agg_result.get('avg_actual_y_perf_of_selected')
                    if avg_perf is not None and np.isfinite(avg_perf):
                        for step in range(expected_length):
                            budget_percent = (step + 1) / expected_length * max_percent
                            combined_data.append({
                                'Method': fixed_method,
                                'Budget_Step': step + 1, 
                                'Budget_Percent': budget_percent,
                                'Avg_Performance': avg_perf
                            })
            
            if combined_data:
                combined_df = pd.DataFrame(combined_data)
                combined_csv_path = os.path.join(csv_dir, f'all_methods_trajectories_{args.num_runs}runs.csv')
                combined_df.to_csv(combined_csv_path, index=False)
                print(f"Saved combined trajectories data to: {combined_csv_path}")
                
                summary_data = []
                for method in set(combined_df['Method']):
                    method_data = combined_df[combined_df['Method'] == method]
                    if method not in ['Actual Best (Overall Dataset)', 'Full Graph']:
                        max_step_data = method_data[method_data['Budget_Step'] == method_data['Budget_Step'].max()]
                        if not max_step_data.empty:
                            final_perf = max_step_data['Avg_Performance'].values[0]
                            
                            method_agg_results = next((item for item in aggregated_results_list if item['method'] == method), None)
                            if method_agg_results:
                                summary_data.append({
                                    'Method': method,
                                    'Final_Performance': final_perf,
                                    'Evaluation_Time': method_agg_results.get('avg_evaluation_time', 0.0),
                                    'Run_Time': method_agg_results.get('avg_run_time', 0.0)
                                })
                            else:
                                summary_data.append({
                                    'Method': method,
                                    'Final_Performance': final_perf,
                                    'Evaluation_Time': 0.0,
                                    'Run_Time': 0.0
                                })
                    else:
                        if not method_data.empty:
                            final_perf = method_data['Avg_Performance'].values[0]
                            
                            method_agg_results = next((item for item in aggregated_results_list if item['method'] == method), None)
                            if method_agg_results:
                                summary_data.append({
                                    'Method': method,
                                    'Final_Performance': final_perf,
                                    'Evaluation_Time': method_agg_results.get('avg_evaluation_time', 0.0),
                                    'Run_Time': method_agg_results.get('avg_run_time', 0.0)
                                })
                            else:
                                summary_data.append({
                                    'Method': method,
                                    'Final_Performance': final_perf,
                                    'Evaluation_Time': 0.0,
                                    'Run_Time': 0.0
                                })
                
                if original_overall_actual_y is not None:
                    perf_values = original_overall_actual_y.cpu().numpy()
                    
                    if higher_is_better:
                        best_perf = float(np.max(perf_values))
                        top3_indices = np.argsort(perf_values)[-3:]
                    else:
                        best_perf = float(np.min(perf_values))
                        top3_indices = np.argsort(perf_values)[:3]
                    
                    top3_perfs = perf_values[top3_indices]
                    avg_top3_perf = float(np.mean(top3_perfs))
                    
                    for item in summary_data:
                        item['Best_Performance'] = best_perf
                        item['Avg_Top3_Performance'] = avg_top3_perf
                
                summary_df = pd.DataFrame(summary_data)
                summary_csv_path = os.path.join(csv_dir, f'performance_summary_{args.num_runs}runs.csv')
                summary_df.to_csv(summary_csv_path, index=False)
                print(f"Saved performance summary to: {summary_csv_path}")
            
        except Exception as e:
            print(f"Error saving trajectory data to CSV: {e}")
            import traceback
            traceback.print_exc()
