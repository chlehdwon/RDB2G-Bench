import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict, Optional, Union, List
from scipy.stats import norm

from ..dataset import PerformancePredictionDataset
from ..micro_action import MicroActionSet
from .utils import calculate_overall_rank, get_performance_for_index, update_trajectory_and_best, pad_trajectory, calculate_evaluation_time

def get_state_embedding(edge_set: Union[list, tuple], embedding_dim: int) -> Optional[np.ndarray]:
    """
    Convert edge set representation into a fixed-size numerical embedding.
    
    This function transforms the binary edge set representation into a padded
    or truncated numpy array suitable for neural network input.
    
    Args:
        edge_set (Union[list, tuple]): Binary representation of which edges are active
            in the current graph configuration
        embedding_dim (int): Target dimensionality for the embedding vector
        
    Returns:
        Optional[np.ndarray]: Fixed-size embedding array of shape (1, embedding_dim),
            or None if edge_set is None
            
    Example:
        >>> edge_set = [1, 0, 1, 0]
        >>> embedding = get_state_embedding(edge_set, 8)
        >>> print(embedding.shape)
        (1, 8)
    """
    if edge_set is None:
        return None
    numeric_edge_set = [int(e) if isinstance(e, bool) else e for e in edge_set]

    embedding = np.zeros(embedding_dim, dtype=np.float32)
    edge_vector = np.array(numeric_edge_set, dtype=np.float32)

    current_len = len(edge_vector)
    if current_len <= embedding_dim:
        embedding[:current_len] = edge_vector
    else:
        embedding = edge_vector[:embedding_dim]

    return embedding.reshape(1, -1)

class MLPSurrogate(nn.Module):
    """
    Multi-Layer Perceptron surrogate model for architecture performance prediction.
    
    This neural network serves as a surrogate model to approximate the performance
    of different graph neural network architectures. It takes graph configuration
    embeddings as input and predicts their expected performance.
    
    Attributes:
        layers (nn.Sequential): Sequential layers of the MLP including dropout
        
    Args:
        input_dim (int): Dimensionality of input embeddings
        hidden_dim1 (int): Size of first hidden layer. Defaults to 64.
        hidden_dim2 (int): Size of second hidden layer. Defaults to 64.
        dropout_rate (float): Dropout probability for regularization. Defaults to 0.1.
        
    Example:
        >>> model = MLPSurrogate(input_dim=20, hidden_dim1=32, hidden_dim2=32)
        >>> x = torch.randn(10, 20)
        >>> predictions = model(x)
        >>> print(predictions.shape)
        torch.Size([10, 1])
    """
    
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 64, dropout_rate: float = 0.1):
        """
        Initialize the MLP surrogate model.
        
        Args:
            input_dim (int): Dimensionality of input embeddings
            hidden_dim1 (int): Size of first hidden layer
            hidden_dim2 (int): Size of second hidden layer 
            dropout_rate (float): Dropout probability for regularization
        """
        super(MLPSurrogate, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the surrogate model.
        
        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Performance predictions of shape (batch_size, 1)
        """
        return self.layers(x)

def bananas_loss(y_pred: torch.Tensor, y_true: torch.Tensor, y_lb: float, eps: float = 1e-9):
    """
    Calculate the BANANAS loss for architecture performance prediction.
    
    The BANANAS loss is designed specifically for neural architecture search
    and is computed as: L = mean | (y_pred - y_lb) / (y_true - y_lb) - 1 |
    where y_lb is the lower bound of performance values.
    
    Args:
        y_pred (torch.Tensor): Predicted performance values
        y_true (torch.Tensor): True performance values
        y_lb (float): Lower bound of performance values for normalization
        eps (float): Small epsilon to prevent division by zero. Defaults to 1e-9.
        
    Returns:
        torch.Tensor: Computed BANANAS loss value
        
    Reference:
        White, Colin, et al. "BANANAS: Bayesian optimization with neural 
        architectures for neural architecture search." AAAI 2021.
    """
    numerator = y_pred - y_lb
    denominator = y_true - y_lb
    loss = torch.mean(torch.abs(numerator / (denominator + eps) - 1.0))
    return loss

def bayesian_optimization_analysis(
    dataset: PerformancePredictionDataset,
    micro_action_set: MicroActionSet,
    overall_actual_y: Optional[torch.Tensor],
    higher_is_better: bool,
    termination_threshold_ratio: float,
    method_name: str = "Bayesian Optimization Heuristic",
    initial_sampling_size: int = 10,
    max_iterations: int = 100,
    mlp_hidden_dim1: int = 32,
    mlp_hidden_dim2: int = 32,
    mlp_dropout_rate: float = 0.1,
    mlp_learning_rate: float = 0.001,
    mlp_epochs_per_iteration: int = 50,
    mlp_batch_size: int = 32,
    ei_mc_samples: int = 50,
):
    """
    Perform Neural Architecture Search using Bayesian Optimization.
    
    This function implements a complete Bayesian optimization loop for finding
    optimal graph neural network architectures. It uses an MLP surrogate model
    to approximate performance and Expected Improvement for acquisition.
    
    Args:
        dataset (PerformancePredictionDataset): Dataset containing architecture 
            performance data
        micro_action_set (MicroActionSet): Set of micro actions for architecture
            space exploration
        overall_actual_y (Optional[torch.Tensor]): Complete performance tensor for
            ranking calculations
        higher_is_better (bool): Whether higher performance values are better
        termination_threshold_ratio (float): Fraction of total architectures to
            evaluate as budget
        method_name (str): Name identifier for this method. 
            Defaults to "Bayesian Optimization Heuristic".
        initial_sampling_size (int): Number of random architectures to evaluate
            initially. Defaults to 10.
        max_iterations (int): Maximum number of optimization iterations.
            Defaults to 100.
        mlp_hidden_dim1 (int): First hidden layer size for MLP surrogate.
            Defaults to 32.
        mlp_hidden_dim2 (int): Second hidden layer size for MLP surrogate.
            Defaults to 32.
        mlp_dropout_rate (float): Dropout rate for MLP surrogate. Defaults to 0.1.
        mlp_learning_rate (float): Learning rate for MLP training. Defaults to 0.001.
        mlp_epochs_per_iteration (int): Training epochs per BO iteration.
            Defaults to 50.
        mlp_batch_size (int): Batch size for MLP training. Defaults to 32.
        ei_mc_samples (int): Monte Carlo samples for uncertainty estimation.
            Defaults to 50.
            
    Returns:
        Dict[str, Union[str, int, float, List, Optional[int]]]: Dictionary containing search results and performance metrics.
        
        - method (str): Method name
        - selected_graph_id (Optional[int]): Index of best found architecture
        - actual_y_perf_of_selected (float): Performance of selected architecture
        - selection_metric_value (float): Metric value used for selection
        - selected_graph_origin (str): Origin method name
        - discovered_count (int): Number of architectures evaluated
        - total_iterations_run (int): Number of BO iterations completed
        - rank_position_overall (float): Rank among all architectures
        - percentile_overall (float): Percentile ranking
        - total_samples_overall (int): Total available architectures
        - performance_trajectory (List): Performance over time
        - total_evaluation_time (float): Time spent on evaluations
        - total_run_time (float): Total algorithm runtime
            
    Example:
        >>> results = bayesian_optimization_analysis(
        ...     dataset=dataset,
        ...     micro_action_set=micro_actions,
        ...     overall_actual_y=y_tensor,
        ...     higher_is_better=True,
        ...     termination_threshold_ratio=0.05,
        ...     max_iterations=50
        ... )
        >>> print(f"Best architecture: {results['selected_graph_id']}")
        >>> print(f"Performance: {results['actual_y_perf_of_selected']:.4f}")
    """
    performance_cache = {}
    embedding_cache = {}
    time_cache = {}
    evaluated_indices = set()
    total_evaluated_count = 0
    performance_trajectory = []
    best_perf_so_far = -np.inf if higher_is_better else np.inf
    best_index_so_far = None
    total_evaluation_time = 0.0
    
    start_time = time.time()

    num_total_valid_graphs = len(micro_action_set.valid_edge_sets_list)
    if num_total_valid_graphs == 0:
        print("Error: MicroActionSet contains no valid graphs. Cannot run BANANAS.")
        return {
            "method": method_name, "selected_graph_id": None, "actual_y_perf_of_selected": np.nan,
            "selection_metric_value": np.nan, "selected_graph_origin": method_name,
            "discovered_count": 0, "total_iterations_run": 0,
            "rank_position_overall": np.nan, "percentile_overall": np.nan,
            "total_samples_overall": overall_actual_y.numel() if overall_actual_y is not None else 0,
            "performance_trajectory": performance_trajectory,
            "total_evaluation_time": 0.0, "total_run_time": 0.0
        }

    if not hasattr(micro_action_set, 'full_edges') or not micro_action_set.full_edges:
         print("Error: micro_action_set missing 'full_edges' attribute or it's empty.")
         return {
            "method": method_name, "selected_graph_id": None, "actual_y_perf_of_selected": np.nan,
            "selection_metric_value": np.nan, "selected_graph_origin": method_name,
            "discovered_count": 0, "total_iterations_run": 0,
            "rank_position_overall": np.nan, "percentile_overall": np.nan,
            "total_samples_overall": overall_actual_y.numel() if overall_actual_y is not None else 0,
            "performance_trajectory": performance_trajectory,
            "total_evaluation_time": 0.0, "total_run_time": 0.0
        }
    embedding_dim = len(micro_action_set.full_edges)

    y_lb = None
    if overall_actual_y is not None and overall_actual_y.numel() > 0:
        y_values_np = overall_actual_y.cpu().numpy()
        y_lb = float(np.min(y_values_np))
    else:
        y_lb = 0.0

    evaluation_budget = max(1, int(termination_threshold_ratio * num_total_valid_graphs))

    X_train_np = []
    y_train_np = []
    initial_indices_to_sample = []

    if num_total_valid_graphs > 0:
        sample_size = min(initial_sampling_size, evaluation_budget, num_total_valid_graphs)
        if sample_size > 0:
            sample_size = min(sample_size, num_total_valid_graphs)
            initial_indices_to_sample = np.random.choice(num_total_valid_graphs, size=sample_size, replace=False)

    for index in initial_indices_to_sample:
        if total_evaluated_count >= evaluation_budget:
             break

        edge_set = micro_action_set.valid_edge_sets_list[index]
        embedding = get_state_embedding(edge_set, embedding_dim)
        initial_cache_size = len(performance_cache)
        perf = get_performance_for_index(index, dataset, performance_cache)

        if perf is not None and embedding is not None:
            X_train_np.append(embedding.flatten())
            y_train_np.append(perf)
            embedding_cache[index] = embedding
            evaluated_indices.add(index)
            
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

    if not X_train_np:
        pad_trajectory(performance_trajectory, total_evaluated_count, evaluation_budget, method_name)
        total_run_time = time.time() - start_time
        return {
            "method": method_name, "selected_graph_id": None, "actual_y_perf_of_selected": np.nan,
            "selection_metric_value": np.nan, "selected_graph_origin": method_name,
            "discovered_count": 0, "total_iterations_run": 0,
            "rank_position_overall": np.nan, "percentile_overall": np.nan,
            "total_samples_overall": overall_actual_y.numel() if overall_actual_y is not None else 0,
            "performance_trajectory": performance_trajectory,
            "total_evaluation_time": total_evaluation_time, "total_run_time": total_run_time
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_surrogate = MLPSurrogate(
        input_dim=embedding_dim,
        hidden_dim1=mlp_hidden_dim1,
        hidden_dim2=mlp_hidden_dim2,
        dropout_rate=mlp_dropout_rate
    ).to(device)
    optimizer = optim.Adam(mlp_surrogate.parameters(), lr=mlp_learning_rate)

    final_iteration_count = 0
    for iteration in range(max_iterations):
        final_iteration_count = iteration + 1
        if total_evaluated_count >= evaluation_budget:
            break

        if len(X_train_np) < mlp_batch_size:
             current_batch_size = len(X_train_np)
        else:
             current_batch_size = mlp_batch_size

        mlp_surrogate.train()
        X_train_tensor = torch.tensor(np.array(X_train_np), dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(np.array(y_train_np), dtype=torch.float32).unsqueeze(1).to(device)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)

        epoch_loss_agg = 0.0
        num_batches_processed = 0

        for epoch in range(mlp_epochs_per_iteration):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                y_pred = mlp_surrogate(batch_x)
                loss = bananas_loss(y_pred, batch_y, y_lb)
                loss.backward()
                optimizer.step()
                epoch_loss_agg += loss.item()
                num_batches_processed +=1

        avg_loss_this_iter = epoch_loss_agg / num_batches_processed if num_batches_processed > 0 else 0
        mlp_surrogate.eval()

        candidate_indices = set()
        if best_index_so_far is not None and best_index_so_far >= 0:
            current_best_edge_set = micro_action_set.valid_edge_sets_list[best_index_so_far]

            if current_best_edge_set:
                action_fns = [
                    micro_action_set.add_fk_pk_edge, micro_action_set.remove_fk_pk_edge,
                    micro_action_set.convert_row_to_edge, micro_action_set.convert_edge_to_row,
                ]
                for action_fn in action_fns:
                     if hasattr(micro_action_set, action_fn.__name__):
                        possible_next_states = action_fn(current_best_edge_set)
                        if possible_next_states:
                            for _, next_index in possible_next_states:
                                if 0 <= next_index < num_total_valid_graphs and next_index not in evaluated_indices:
                                    candidate_indices.add(next_index)

        if not candidate_indices and total_evaluated_count < evaluation_budget:
            remaining_budget = evaluation_budget - total_evaluated_count
            num_random_to_try = min(max(1, int(0.05 * num_total_valid_graphs)),
                                    remaining_budget,
                                    num_total_valid_graphs - len(evaluated_indices))

            if num_random_to_try > 0:
                 all_indices = np.arange(num_total_valid_graphs)
                 evaluated_list = list(evaluated_indices)
                 unevaluated_mask = np.isin(all_indices, evaluated_list, invert=True)
                 available_indices = all_indices[unevaluated_mask]

                 if len(available_indices) > 0:
                      actual_sample_size = min(num_random_to_try, len(available_indices))
                      random_candidates = np.random.choice(available_indices, size=actual_sample_size, replace=False)
                      candidate_indices.update(random_candidates)

        if not candidate_indices:
            break

        candidate_embeddings_list = []
        valid_candidate_indices = []
        for index in candidate_indices:
            if total_evaluated_count + (len(valid_candidate_indices) + 1) > evaluation_budget + max_iterations:
                 pass

            embedding = embedding_cache.get(index)
            if embedding is None:
                edge_set = micro_action_set.valid_edge_sets_list[index]
                embedding = get_state_embedding(edge_set, embedding_dim)
                if embedding is not None: embedding_cache[index] = embedding

            if embedding is not None:
                candidate_embeddings_list.append(embedding)
                valid_candidate_indices.append(index)

        if not valid_candidate_indices:
            continue

        candidate_means = []
        candidate_sigmas = []
        mlp_surrogate.train()
        with torch.no_grad():
            for embedding_np in candidate_embeddings_list:
                embedding_tensor = torch.tensor(embedding_np, dtype=torch.float32).to(device)
                mc_predictions = []
                if embedding_tensor.ndim == 1: embedding_tensor = embedding_tensor.unsqueeze(0)
                if embedding_tensor.shape[1] != embedding_dim:
                    candidate_means.append(np.nan)
                    candidate_sigmas.append(np.nan)
                    continue

                for _ in range(ei_mc_samples):
                    pred = mlp_surrogate(embedding_tensor)
                    mc_predictions.append(pred.item())

                valid_mc_preds = [p for p in mc_predictions if not np.isnan(p)]
                if len(valid_mc_preds) > 0:
                    candidate_means.append(np.mean(valid_mc_preds))
                    candidate_sigmas.append(np.std(valid_mc_preds))
                else:
                     candidate_means.append(np.nan)
                     candidate_sigmas.append(np.nan)

        mlp_surrogate.eval()

        valid_results_mask = ~np.isnan(candidate_means)
        if not np.any(valid_results_mask):
             continue

        mu_candidates = np.array(candidate_means)[valid_results_mask]
        sigma_candidates = np.array(candidate_sigmas)[valid_results_mask]
        valid_candidate_indices_after_mc = [idx for i, idx in enumerate(valid_candidate_indices) if valid_results_mask[i]]

        f_best = best_perf_so_far

        sigma_candidates = np.maximum(sigma_candidates, 1e-9)

        if higher_is_better:
            improvement = mu_candidates - f_best
            Z = improvement / sigma_candidates
            ei_scores = improvement * norm.cdf(Z) + sigma_candidates * norm.pdf(Z)
        else:
            improvement = f_best - mu_candidates
            Z = improvement / sigma_candidates
            ei_scores = improvement * norm.cdf(Z) + sigma_candidates * norm.pdf(Z)

        ei_scores = np.nan_to_num(ei_scores, nan=-np.inf)

        best_candidate_idx_in_filtered_list = np.argmax(ei_scores)
        next_index_to_evaluate = valid_candidate_indices_after_mc[best_candidate_idx_in_filtered_list]

        if total_evaluated_count >= evaluation_budget:
             break

        next_edge_set = micro_action_set.valid_edge_sets_list[next_index_to_evaluate]

        next_embedding = embedding_cache.get(next_index_to_evaluate)
        if next_embedding is None: next_embedding = get_state_embedding(next_edge_set, embedding_dim)

        initial_cache_size = len(performance_cache)
        next_perf = get_performance_for_index(next_index_to_evaluate, dataset, performance_cache)

        evaluated_indices.add(next_index_to_evaluate)

        if next_embedding is not None and next_perf is not None:
            X_train_np.append(next_embedding.flatten())
            y_train_np.append(next_perf)
            
            performance_cache[next_index_to_evaluate] = next_perf
            
            eval_time = calculate_evaluation_time(next_index_to_evaluate, dataset, time_cache)
            if eval_time is not None:
                if len(performance_cache) > initial_cache_size:
                    total_evaluation_time += eval_time
            
            total_evaluated_count, best_perf_so_far, best_index_so_far = \
                update_trajectory_and_best(
                    next_index_to_evaluate, next_perf, performance_cache, initial_cache_size,
                    total_evaluated_count, performance_trajectory,
                    best_perf_so_far, best_index_so_far, higher_is_better
                )

    pad_trajectory(performance_trajectory, total_evaluated_count, evaluation_budget, method_name)

    total_run_time = time.time() - start_time

    final_selected_index = best_index_so_far if best_index_so_far is not None and best_index_so_far >= 0 else None
    final_selected_perf = np.nan
    if final_selected_index is not None and final_selected_index in performance_cache:
         final_selected_perf = performance_cache[final_selected_index]
    elif final_selected_index is not None:
         final_selected_perf = best_perf_so_far

    results = {
        "method": method_name,
        "selected_graph_id": final_selected_index,
        "actual_y_perf_of_selected": final_selected_perf if final_selected_index is not None else np.nan,
        "selection_metric_value": final_selected_perf if final_selected_index is not None else np.nan,
        "selected_graph_origin": method_name,
        "discovered_count": len(performance_cache),
        "total_iterations_run": final_iteration_count,
        "rank_position_overall": np.nan,
        "percentile_overall": np.nan,
        "total_samples_overall": overall_actual_y.numel() if overall_actual_y is not None else 0,
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
