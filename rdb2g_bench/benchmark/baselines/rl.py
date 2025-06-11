import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
import time
import pandas as pd
from collections import deque
from typing import Union, Tuple, Optional, List

from ..dataset import PerformancePredictionDataset
from ..micro_action import MicroActionSet
from .utils import calculate_overall_rank, get_performance_for_index, update_trajectory_and_best, pad_trajectory, calculate_evaluation_time

class ControllerRNN(nn.Module):
    """
    Recurrent Neural Network Controller for Architecture Search.
    
    This controller generates sequences of actions to construct graph neural network
    architectures. It uses either LSTM or GRU cells to maintain state across the
    action sequence and outputs action probabilities for policy gradient training.
    
    Attributes:
        hidden_dim (int): Hidden dimension of the RNN
        num_layers (int): Number of RNN layers
        rnn_type (str): Type of RNN cell ('lstm' or 'gru')
        rnn (nn.Module): The RNN module (LSTM or GRU)
        action_head (nn.Linear): Linear layer for action probability output
    
    Args:
        input_dim (int): Dimensionality of input state embeddings
        hidden_dim (int): Hidden dimension of the RNN
        num_actions (int): Number of possible actions in the action space
        rnn_type (str): Type of RNN to use ('lstm' or 'gru'). Defaults to 'lstm'.
        num_layers (int): Number of RNN layers. Defaults to 1.
        
    Example:
        >>> controller = ControllerRNN(
        ...     input_dim=20,
        ...     hidden_dim=64,
        ...     num_actions=10,
        ...     rnn_type='lstm'
        ... )
        >>> state_emb = torch.randn(1, 1, 20)
        >>> hidden = controller.init_hidden()
        >>> action_logits, new_hidden = controller(state_emb, hidden)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int, rnn_type: str = 'lstm', num_layers: int = 1):
        """
        Initialize the Controller RNN.
        
        Args:
            input_dim (int): Dimensionality of input state embeddings
            hidden_dim (int): Hidden dimension of the RNN
            num_actions (int): Number of possible actions
            rnn_type (str): Type of RNN ('lstm' or 'gru')
            num_layers (int): Number of RNN layers
        """
        super(ControllerRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Choose 'lstm' or 'gru'.")

        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, state_embedding: torch.Tensor, hidden_state: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
        """
        Forward pass through the controller.
        
        Args:
            state_embedding (torch.Tensor): Current state embedding of shape (batch, seq, input_dim)
            hidden_state (tuple or torch.Tensor): Hidden state from previous step
            
        Returns:
            tuple: (action_logits, new_hidden_state)
                - action_logits: Logits for action selection of shape (batch, num_actions)
                - new_hidden_state: Updated hidden state
        """
        output, next_hidden = self.rnn(state_embedding, hidden_state)
        action_logits = self.action_head(output.squeeze(1))
        return action_logits, next_hidden

    def init_hidden(self, batch_size: int = 1) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Initialize hidden state for the RNN.
        
        Args:
            batch_size (int): Batch size for initialization
            
        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: Initial hidden state.
                For LSTM: tuple of (hidden_state, cell_state) tensors of shape 
                (num_layers, batch_size, hidden_dim).
                For GRU: single tensor of shape (num_layers, batch_size, hidden_dim).
        """
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                    weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        elif self.rnn_type == 'gru':
            return weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()
        else:
             raise ValueError(f"Trying to initialize hidden state for unsupported RNN type: {self.rnn_type}")

def get_state_embedding(current_edge_set: Union[tuple, list], embedding_dim: int) -> torch.Tensor:
    """
    Convert edge set representation into a fixed-size embedding for RNN input.
    
    This function transforms the binary edge set representation into a padded
    or truncated tensor suitable for the RNN controller input.
    
    Args:
        current_edge_set (tuple or list): Binary representation of active edges
        embedding_dim (int): Target dimensionality for the embedding
        
    Returns:
        torch.Tensor: State embedding of shape (1, 1, embedding_dim)
        
    Example:
        >>> edge_set = (1, 0, 1, 0)
        >>> embedding = get_state_embedding(edge_set, 8)
        >>> print(embedding.shape)
        torch.Size([1, 1, 8])
    """
    embedding = torch.zeros(1, 1, embedding_dim)
    edge_tensor = torch.tensor(current_edge_set, dtype=torch.float)
    current_len = len(edge_tensor)
    if current_len < embedding_dim:
        padding = torch.zeros(embedding_dim - current_len)
        edge_tensor = torch.cat((edge_tensor, padding))
    elif current_len > embedding_dim:
        edge_tensor = edge_tensor[:embedding_dim]
    embedding[0, 0, :] = edge_tensor
    return embedding

def get_action_space(micro_action_set: MicroActionSet, current_edge_set: Union[tuple, list]) -> Tuple[List[Tuple[dict, int]], int]:
    """
    Determine valid actions from the current architecture state.
    
    This function analyzes the current edge set and returns all possible
    micro actions that can be applied, along with a special stop action.
    
    Args:
        micro_action_set (MicroActionSet): Set of available micro actions
        current_edge_set (tuple): Current binary edge set representation
        
    Returns:
        tuple: (possible_actions_with_ids, num_total_valid_actions)
            - possible_actions_with_ids: List of (action_detail, action_id) tuples
            - num_total_valid_actions: Total number of available actions including stop
            
    Example:
        >>> actions, num_actions = get_action_space(micro_actions, current_set)
        >>> print(f"Available actions: {num_actions}")
    """
    possible_actions_with_ids = []
    action_id_counter = 0
    num_total_valid_graphs = len(micro_action_set.valid_edge_sets_list)

    action_methods_to_call = [
        ("add_fk_pk", micro_action_set.add_fk_pk_edge),
        ("remove_fk_pk", micro_action_set.remove_fk_pk_edge),
        ("convert_n2e", micro_action_set.convert_row_to_edge),
        ("convert_e2n", micro_action_set.convert_edge_to_row)
    ]
    available_action_methods = [
        (name, func) for name, func in action_methods_to_call if hasattr(micro_action_set, func.__name__)
    ]

    for action_type, action_fn in available_action_methods:
        possible_next_states = action_fn(current_edge_set)
        if possible_next_states:
            for _, next_index in possible_next_states:
                if 0 <= next_index < num_total_valid_graphs:
                    action_detail = {"type": action_type, "target_index": next_index}
                    possible_actions_with_ids.append((action_detail, action_id_counter))
                    action_id_counter += 1

    stop_action_detail = {"type": "stop", "target_index": -1}
    possible_actions_with_ids.append((stop_action_detail, action_id_counter))
    num_total_valid_actions = action_id_counter + 1

    if num_total_valid_actions <= 1:
        current_idx_for_warning = -1
        try:
            current_idx_for_warning = micro_action_set.valid_edge_sets_list.index(current_edge_set)
        except ValueError: pass
        if not any(a[0]["type"] != "stop" for a in possible_actions_with_ids):
             print(f"Warning: Only STOP action available from current state index {current_idx_for_warning}.")

    return possible_actions_with_ids, num_total_valid_actions

def apply_action(current_edge_set: Union[tuple, list], action_details: dict, micro_action_set: MicroActionSet) -> Tuple[Union[tuple, list], int]:
    """
    Apply the selected action to transition to a new architecture state.
    
    This function executes the chosen micro action and returns the resulting
    edge set configuration and its index in the valid edge sets list.
    
    Args:
        current_edge_set (tuple): Current binary edge set representation
        action_details (dict): Action specification with 'type' and 'target_index'
        micro_action_set (MicroActionSet): Set of available micro actions
        
    Returns:
        tuple: (new_edge_set, new_index)
            - new_edge_set: Resulting edge set after action application
            - new_index: Index of new edge set in valid_edge_sets_list
            
    Example:
        >>> action = {"type": "add_fk_pk", "target_index": 42}
        >>> new_set, new_idx = apply_action(current_set, action, micro_actions)
    """
    target_index = action_details.get("target_index")
    action_type = action_details.get("type")

    if action_type == "stop" or target_index == -1:
        current_index = -1
        try:
            current_index = micro_action_set.valid_edge_sets_list.index(current_edge_set)
        except ValueError: pass
        return current_edge_set, current_index

    if target_index is not None and 0 <= target_index < len(micro_action_set.valid_edge_sets_list):
        next_index = target_index
        next_edge_set = micro_action_set.valid_edge_sets_list[next_index]
        return next_edge_set, next_index
    else:
        print(f"Error: Invalid target_index ({target_index}) received in apply_action. Returning current state.")
        current_index = -1
        try:
            current_index = micro_action_set.valid_edge_sets_list.index(current_edge_set)
        except ValueError: pass
        return current_edge_set, current_index

def rl_heuristic_analysis(
    dataset: PerformancePredictionDataset,
    micro_action_set: MicroActionSet,
    overall_actual_y: torch.Tensor,
    higher_is_better: bool,
    controller_rnn_type: str = 'lstm',
    controller_num_layers: int = 1,
    controller_hidden_dim: int = 32,
    learning_rate: float = 0.005,
    num_episodes: int = 50,
    max_steps_per_episode: int = 5,
    gamma: float = 0.99,
    termination_threshold_ratio: float = 0.1,
    method_name: str = "RL Heuristic (Policy Gradient)",
) -> dict:
    """
    Perform Neural Architecture Search using Reinforcement Learning with Policy Gradients.
    
    This function implements a complete RL-based search using REINFORCE algorithm.
    An RNN controller learns to generate sequences of micro actions that construct
    high-performing graph neural network architectures.
    
    Args:
        dataset (PerformancePredictionDataset): Dataset containing architecture 
            performance data
        micro_action_set (MicroActionSet): Set of micro actions for architecture
            space exploration
        overall_actual_y (torch.Tensor): Complete performance tensor for
            ranking calculations
        higher_is_better (bool): Whether higher performance values are better
        controller_rnn_type (str): RNN type for controller ('lstm' or 'gru').
            Defaults to 'lstm'.
        controller_num_layers (int): Number of RNN layers. Defaults to 1.
        controller_hidden_dim (int): Hidden dimension of controller RNN.
            Defaults to 32.
        learning_rate (float): Learning rate for controller optimization.
            Defaults to 0.005.
        num_episodes (int): Number of RL episodes to run. Defaults to 50.
        max_steps_per_episode (int): Maximum steps per episode. Defaults to 5.
        gamma (float): Discount factor for returns computation. Defaults to 0.99.
        termination_threshold_ratio (float): Fraction of total architectures to
            evaluate as budget. Defaults to 0.1.
        method_name (str): Name identifier for this method. 
            Defaults to "RL Heuristic (Policy Gradient)".
            
    Returns:
        Dict[str, Union[str, int, float, List, Optional[int]]]: Dictionary containing search results and performance metrics.
        
        - method (str): Method name
        - selected_graph_id (Optional[int]): Index of best found architecture
        - actual_y_perf_of_selected (float): Performance of selected architecture
        - selection_metric_value (float): Metric value used for selection
        - selected_graph_origin (str): Origin method name
        - discovered_count (int): Number of architectures evaluated
        - total_iterations_run (int): Number of episodes completed
        - rank_position_overall (float): Rank among all architectures
        - percentile_overall (float): Percentile ranking
        - total_samples_overall (int): Total available architectures
        - performance_trajectory (List): Performance over time
        - total_evaluation_time (float): Time spent on evaluations
        - total_run_time (float): Total algorithm runtime
            
    Example:
        >>> results = rl_heuristic_analysis(
        ...     dataset=dataset,
        ...     micro_action_set=micro_actions,
        ...     overall_actual_y=y_tensor,
        ...     higher_is_better=True,
        ...     controller_rnn_type='lstm',
        ...     controller_hidden_dim=64,
        ...     num_episodes=100,
        ...     max_steps_per_episode=10
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
    global_best_architecture_sequence = []
    method_origin = "RL Heuristic (Policy Gradient)"
    total_samples_overall = overall_actual_y.numel() if overall_actual_y is not None else 0
    total_evaluation_time = 0.0
    
    start_time = time.time()

    if num_total_valid_graphs == 0:
        print("Error: MicroActionSet contains no valid graphs. Cannot run RL.")
        return {
            "method": method_name, "selected_graph_id": None, "actual_y_perf_of_selected": np.nan,
            "selection_metric_value": np.nan, "selected_graph_origin": method_origin,
            "discovered_count": 0, "total_iterations_run": 0,
            "rank_position_overall": np.nan, "percentile_overall": np.nan,
            "total_samples_overall": total_samples_overall, "performance_trajectory": [],
            "total_evaluation_time": 0.0, "total_run_time": time.time() - start_time
        }

    evaluation_budget = int(termination_threshold_ratio * num_total_valid_graphs)
    evaluation_budget = max(1, min(evaluation_budget, num_total_valid_graphs))
    print(f"{method_name}: Budget set to {evaluation_budget} unique evaluations.")

    if not hasattr(micro_action_set, 'full_edges') or not micro_action_set.full_edges:
         print("Error: micro_action_set missing 'full_edges' attribute or it's empty.")
         return {
            "method": method_name, "selected_graph_id": None, "actual_y_perf_of_selected": np.nan,
            "selection_metric_value": np.nan, "selected_graph_origin": method_origin,
            "discovered_count": 0, "total_iterations_run": 0,
            "rank_position_overall": np.nan, "percentile_overall": np.nan,
            "total_samples_overall": total_samples_overall, "performance_trajectory": [],
            "total_evaluation_time": 0.0, "total_run_time": time.time() - start_time
         }
    state_embedding_dim = len(micro_action_set.full_edges)

    try:
         _, max_possible_actions_estimation = get_action_space(micro_action_set, micro_action_set.valid_edge_sets_list[0])
         print(f"Estimated max actions based on state 0: {max_possible_actions_estimation}")
    except Exception as e:
         max_possible_actions_estimation = state_embedding_dim * 2 + 10
         print(f"Warning: Could not estimate max actions from state 0 ({e}). Using fallback: {max_possible_actions_estimation}")

    controller = ControllerRNN(
        state_embedding_dim, controller_hidden_dim, max_possible_actions_estimation,
        rnn_type=controller_rnn_type, num_layers=controller_num_layers
    )
    optimizer = optim.Adam(controller.parameters(), lr=learning_rate)

    print(f"{method_name}: Starting search with {num_episodes} episodes.")
    print(f"Controller using {controller_rnn_type.upper()} ({controller_num_layers} layers)")

    final_episode_count = 0

    for episode in range(num_episodes):
        final_episode_count = episode + 1
        if total_evaluated_count >= evaluation_budget:
            print(f"Episode {episode+1}: Evaluation budget reached ({total_evaluated_count}/{evaluation_budget}). Stopping training.")
            break

        if num_total_valid_graphs <= 0: break
        start_index = random.randint(0, num_total_valid_graphs - 1)
        current_index = start_index
        current_edge_set = micro_action_set.valid_edge_sets_list[current_index]

        log_probs = []
        rewards = []
        actions_taken = []
        indices_visited = [current_index]
        hidden_state = controller.init_hidden()

        initial_perf = None
        if current_index not in performance_cache and total_evaluated_count < evaluation_budget:
            initial_cache_size = len(performance_cache)
            initial_perf = get_performance_for_index(current_index, dataset, performance_cache)
            
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
            if global_best_index == current_index and initial_perf is not None:
                 global_best_architecture_sequence = []

        elif current_index in performance_cache:
            initial_perf = performance_cache[current_index]
            _, global_best_perf, global_best_index = \
                update_trajectory_and_best(
                     current_index, initial_perf, performance_cache, len(performance_cache),
                     total_evaluated_count, performance_trajectory,
                     global_best_perf, global_best_index, higher_is_better
                )

        last_perf = initial_perf

        for step in range(max_steps_per_episode):
            if total_evaluated_count >= evaluation_budget:
                break

            state_embedding = get_state_embedding(current_edge_set, state_embedding_dim)
            action_logits, hidden_state = controller(state_embedding, hidden_state)
            possible_actions_with_ids, num_valid_actions = get_action_space(micro_action_set, current_edge_set)

            if not possible_actions_with_ids or num_valid_actions <= 1:
                 break

            valid_action_indices = [aid for _, aid in possible_actions_with_ids]
            valid_action_indices_in_bounds = [idx for idx in valid_action_indices if idx < action_logits.size(-1)]

            if not valid_action_indices_in_bounds:
                 print(f"Warning: Ep {episode+1}, Step {step+1}: No valid action indices in controller output bounds. Skipping.")
                 break

            valid_indices_tensor = torch.tensor(valid_action_indices_in_bounds, dtype=torch.long, device=action_logits.device)
            valid_logits = action_logits.gather(1, valid_indices_tensor.unsqueeze(0))
            valid_action_probs = F.softmax(valid_logits, dim=-1)

            try:
                 valid_action_probs = valid_action_probs / valid_action_probs.sum()
                 valid_dist = Categorical(probs=valid_action_probs)
                 sampled_valid_idx = valid_dist.sample().item()
            except RuntimeError as e:
                 print(f"Warning: Sampling failed in Ep {episode+1}, Step {step+1}: {e}. Skipping.")
                 break

            sampled_action_id = valid_action_indices_in_bounds[sampled_valid_idx]
            chosen_action_details = next((details for details, aid in possible_actions_with_ids if aid == sampled_action_id), None)

            if chosen_action_details is None:
                 print(f"Error: Logic error finding action details for ID {sampled_action_id}. Skipping step.")
                 break

            log_prob = valid_dist.log_prob(torch.tensor(sampled_valid_idx))

            if chosen_action_details.get("type") == "stop":
                break

            next_edge_set, next_index = apply_action(current_edge_set, chosen_action_details, micro_action_set)

            step_reward = 0.0
            current_perf = None

            if next_index == current_index:
                 step_reward = 0.0
            elif next_index in indices_visited:
                 step_reward = -0.01
            else:
                 indices_visited.append(next_index)
                 is_newly_evaluated = next_index not in performance_cache

                 if is_newly_evaluated and total_evaluated_count < evaluation_budget:
                     initial_cache_size = len(performance_cache)
                     current_perf = get_performance_for_index(next_index, dataset, performance_cache)
                     
                     if current_perf is not None:
                         performance_cache[next_index] = current_perf
                         
                         eval_time = calculate_evaluation_time(next_index, dataset, time_cache)
                         if eval_time is not None:
                             if len(performance_cache) > initial_cache_size:
                                 total_evaluation_time += eval_time
                         
                     new_total_evaluated_count, new_global_best_perf, new_global_best_index = \
                         update_trajectory_and_best(
                             next_index, current_perf, performance_cache, initial_cache_size,
                             total_evaluated_count, performance_trajectory,
                             global_best_perf, global_best_index, higher_is_better
                         )
                     total_evaluated_count = new_total_evaluated_count
                     if new_global_best_index == next_index and current_perf is not None:
                          global_best_perf = new_global_best_perf
                          global_best_index = new_global_best_index
                          global_best_architecture_sequence = actions_taken + [chosen_action_details]
                     elif global_best_index != new_global_best_index:
                          global_best_perf = new_global_best_perf
                          global_best_index = new_global_best_index

                 elif next_index in performance_cache:
                     current_perf = performance_cache[next_index]
                     _, new_global_best_perf, new_global_best_index = \
                          update_trajectory_and_best(
                              next_index, current_perf, performance_cache, len(performance_cache),
                              total_evaluated_count, performance_trajectory,
                              global_best_perf, global_best_index, higher_is_better
                          )
                     if global_best_index != new_global_best_index:
                          global_best_perf = new_global_best_perf
                          global_best_index = new_global_best_index

                 if current_perf is not None and last_perf is not None and np.isfinite(current_perf) and np.isfinite(last_perf):
                     improvement = current_perf - last_perf
                     step_reward = improvement if higher_is_better else -improvement
                     if (higher_is_better and current_perf is not None and current_perf >= global_best_perf) or \
                        (not higher_is_better and current_perf is not None and current_perf <= global_best_perf):
                         step_reward += 0.05
                 elif current_perf is not None:
                     pass

                 current_edge_set = next_edge_set
                 current_index = next_index
                 last_perf = current_perf

            rewards.append(torch.tensor(float(step_reward)))
            log_probs.append(log_prob)
            actions_taken.append(chosen_action_details)

        if not log_probs or not rewards: continue

        discounted_rewards = []
        R = 0.0
        float_rewards = [r.item() for r in rewards]
        for r in reversed(float_rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards_tensor = torch.tensor(discounted_rewards, dtype=torch.float32)
        if len(discounted_rewards) > 1:
             mean_reward = torch.mean(discounted_rewards_tensor)
             std_reward = torch.std(discounted_rewards_tensor)
             discounted_rewards_tensor = (discounted_rewards_tensor - mean_reward) / (std_reward + 1e-9)

        policy_loss = []
        if len(log_probs) != len(discounted_rewards_tensor):
             print(f"Warning: Mismatch log_probs ({len(log_probs)}) vs rewards ({len(discounted_rewards_tensor)}) in Ep {episode+1}. Skipping update.")
             continue

        for log_prob, reward in zip(log_probs, discounted_rewards_tensor):
             if not isinstance(log_prob, torch.Tensor): log_prob = torch.tensor(log_prob)
             if log_prob.numel() > 1: log_prob = log_prob.flatten()[0]
             if not isinstance(reward, torch.Tensor): reward = torch.tensor(reward)
             if reward.numel() > 1: reward = reward.flatten()[0]
             policy_loss.append(-log_prob * reward.detach())

        if not policy_loss: continue

        optimizer.zero_grad()
        policy_loss_tensor = torch.stack(policy_loss).sum()
        if policy_loss_tensor.requires_grad:
             policy_loss_tensor.backward()
             optimizer.step()

        if (episode + 1) % 20 == 0:
            loss_val = policy_loss_tensor.item() if policy_loss_tensor.requires_grad else "N/A"
            best_perf_val = f"{global_best_perf:.4f}" if global_best_index != -1 else "N/A"
            print(f"Episode {episode+1}/{num_episodes}: Loss: {loss_val}, Best Perf: {best_perf_val}, Budget Used: {total_evaluated_count}/{evaluation_budget}")

    pad_trajectory(performance_trajectory, total_evaluated_count, evaluation_budget, method_name)

    final_selected_index = global_best_index
    final_selected_perf = performance_cache.get(global_best_index, global_best_perf) if global_best_index != -1 else np.nan

    print(f"\n{method_name} Search finished after {final_episode_count} episodes.")
    print(f"Total unique graphs evaluated (budget used): {total_evaluated_count}")
    if final_selected_index != -1:
         print(f"Final Selected Graph (Best Overall): Index {final_selected_index}, Performance: {final_selected_perf:.4f}")
    else:
         print("Final Selected Graph (Best Overall): None found or evaluated.")

    results = {
        "method": method_name,
        "selected_graph_id": final_selected_index if final_selected_index != -1 else None,
        "actual_y_perf_of_selected": final_selected_perf,
        "selection_metric_value": final_selected_perf,
        "selected_graph_origin": method_origin,
        "discovered_count": total_evaluated_count,
        "total_iterations_run": final_episode_count,
        "rank_position_overall": np.nan,
        "percentile_overall": np.nan,
        "total_samples_overall": total_samples_overall,
        "performance_trajectory": performance_trajectory,
        "total_evaluation_time": total_evaluation_time,
        "total_run_time": time.time() - start_time
    }

    if final_selected_index != -1 and overall_actual_y is not None and not np.isnan(final_selected_perf):
        rank_info = calculate_overall_rank(
            final_selected_perf,
            overall_actual_y,
            higher_is_better
        )
        if rank_info:
            results["rank_position_overall"] = rank_info["rank_position_overall"]
            results["percentile_overall"] = rank_info["percentile_overall"]

    return results