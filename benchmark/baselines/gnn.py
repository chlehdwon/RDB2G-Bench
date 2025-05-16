import torch
import time
import copy
import functools
import numpy as np
from scipy.stats import pearsonr, kendalltau

import torch.nn as nn
import torch.optim as optim

from benchmark.baselines.model.gnn_model import PerformancePredictorGNN
from benchmark.baselines.utils import calculate_overall_rank

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []  
    all_targets = []    
    all_graph_ids = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if not hasattr(batch, 'y') or batch.y is None:
                 print(f"Warning: Skipping batch in evaluation due to missing 'y' attribute. Batch keys: {batch.keys}")
                 continue

            out = model(batch)

            target_y = batch.y.view_as(out)
            loss = criterion(out, target_y)
            total_loss += loss.item() * batch.num_graphs
            all_targets.append(batch.y.cpu())

            all_preds.append(out.cpu())
            all_graph_ids.append(batch.graph_id.cpu())

    num_valid_loss_graphs = sum(b.num_graphs for b in loader if hasattr(b, 'y') and b.y is not None)
    avg_loss = total_loss / num_valid_loss_graphs if num_valid_loss_graphs > 0 else 0.0

    pred_y_tensor = torch.cat(all_preds, dim=0)         
    actual_y_tensor = torch.cat(all_targets, dim=0)
    graph_ids_tensor = torch.cat(all_graph_ids, dim=0)

    return avg_loss, pred_y_tensor, actual_y_tensor, graph_ids_tensor

def train(args, perf_pred_dataset, loaders, task_type, device):
    print("\n--- GNN Training and Evaluation Required --- ")
    model = PerformancePredictorGNN(
        num_node_types=perf_pred_dataset.num_nodes,
        task_type=task_type,
        src_node_idx=perf_pred_dataset.src_node_idx,
        dst_node_idx=perf_pred_dataset.dst_node_idx,
        embedding_dim=args.embedding_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        output_dim=1, gnn_type=args.gnn_type, gat_heads=args.gat_heads, dropout_rate=args.dropout_rate,
        embedding_type=args.embedding_type, pooling_type=args.pooling_type, alpha=args.alpha
    ).to(device)
    print("Model initialized.")

    if args.loss_type == "mse":
        criterion = nn.MSELoss()
    elif args.loss_type == "l1":
        criterion = nn.L1Loss()
    elif args.loss_type == "margin":
        criterion = functools.partial(model.margin_loss, margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Starting training...")
    start_time = time.time()
    if args.valid_criterion == "loss":
            best_valid_criterion = float('inf')
    else:
            best_valid_criterion = float('-inf')

    epochs_no_improve = 0
    best_model_state_dict = None
    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0; num_train_samples = 0
        for batch in loaders['train']:
            batch = batch.to(device);
            optimizer.zero_grad();
            out = model(batch)
            target_y = batch.y.view_as(out);
            loss = criterion(out, target_y)
            loss.backward();
            optimizer.step()
            epoch_train_loss += loss.item() * len(batch.y);
            num_train_samples += len(batch.y)
        avg_train_loss = epoch_train_loss / num_train_samples if num_train_samples > 0 else 0

        valid_loss_epoch, valid_pred_y_epoch, valid_actual_y_epoch, _ = evaluate(model, loaders['valid'], criterion, device)
        elapsed_time = time.time() - start_time
        if epoch % 20 == 0 or epoch == args.epochs - 1: print(f'Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {valid_loss_epoch:.4f} | Time: {elapsed_time:.2f}s')

        if args.valid_criterion == "loss":
            if valid_loss_epoch < best_valid_criterion:
                best_valid_criterion = valid_loss_epoch; epochs_no_improve = 0; best_model_state_dict = copy.deepcopy(model.state_dict())
        elif args.valid_criterion == "pearson":
            valid_pearson_epoch, _ = pearsonr(valid_pred_y_epoch.squeeze(), valid_actual_y_epoch)
            if valid_pearson_epoch > best_valid_criterion:
                best_valid_criterion = valid_pearson_epoch; epochs_no_improve = 0; best_model_state_dict = copy.deepcopy(model.state_dict())
        elif args.valid_criterion == "kendall":
            valid_kendall_epoch, _ = kendalltau(valid_pred_y_epoch.squeeze(), valid_actual_y_epoch)
            if valid_kendall_epoch > best_valid_criterion:
                best_valid_criterion = valid_kendall_epoch; epochs_no_improve = 0; best_model_state_dict = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs.")
            break

    print("Training finished.")

    if best_model_state_dict is not None: model.load_state_dict(best_model_state_dict)
    else: print("\nWarning: No best model state saved. Using final model state.")

    return model, criterion

def analyze_predicted_best(
    train_pred_y: torch.Tensor, train_actual_y: torch.Tensor, train_graph_ids: torch.Tensor,
    valid_pred_y: torch.Tensor, valid_actual_y: torch.Tensor, valid_graph_ids: torch.Tensor,
    test_pred_y: torch.Tensor, test_actual_y: torch.Tensor, test_graph_ids: torch.Tensor,
    overall_actual_y: torch.Tensor,
    higher_is_better: bool,
    method_name: str = "GNN Predicted Best",
    predict: str = "test"
):
    def _prepare_tensor(t):
        if t.numel() == 0: return t
        t = t.squeeze()
        return t.unsqueeze(0) if t.ndim == 0 else t

    train_pred_y = _prepare_tensor(train_pred_y)
    train_actual_y = _prepare_tensor(train_actual_y)
    train_graph_ids = _prepare_tensor(train_graph_ids)
    valid_pred_y = _prepare_tensor(valid_pred_y)
    valid_actual_y = _prepare_tensor(valid_actual_y)
    valid_graph_ids = _prepare_tensor(valid_graph_ids)
    test_pred_y = _prepare_tensor(test_pred_y)
    test_actual_y = _prepare_tensor(test_actual_y)
    test_graph_ids = _prepare_tensor(test_graph_ids)

    if predict == "test":
        combined_selection_values = torch.cat([train_pred_y, valid_pred_y, test_pred_y], dim=0)
        combined_actual_y_for_selection = torch.cat([train_actual_y, valid_actual_y, test_actual_y], dim=0)
        combined_graph_ids_for_selection = torch.cat([train_graph_ids, valid_graph_ids, test_graph_ids], dim=0)
    elif predict == "all":
        combined_selection_values = torch.cat([train_actual_y, valid_actual_y, test_pred_y], dim=0)
        combined_actual_y_for_selection = torch.cat([train_actual_y, valid_actual_y, test_actual_y], dim=0)
        combined_graph_ids_for_selection = torch.cat([train_graph_ids, valid_graph_ids, test_graph_ids], dim=0)
    elif predict == "gt":
        k = max(1, (len(train_actual_y) + len(valid_actual_y) + len(test_actual_y)) // 100)
        k = min(k, len(test_pred_y))

        top_k = torch.topk(test_pred_y, k=k, largest=higher_is_better)

        combined_selection_values = torch.cat([train_actual_y, valid_actual_y, test_actual_y[top_k.indices]], dim=0)
        combined_actual_y_for_selection = torch.cat([train_actual_y, valid_actual_y, test_actual_y[top_k.indices]], dim=0)
        combined_graph_ids_for_selection = torch.cat([train_graph_ids, valid_graph_ids, test_graph_ids[top_k.indices]], dim=0)

    if higher_is_better:
        combined_best_idx = torch.argmax(combined_selection_values).item()
    else:
        combined_best_idx = torch.argmin(combined_selection_values).item()

    results = {"method": method_name}
    results["selected_graph_id"] = combined_graph_ids_for_selection[combined_best_idx].item()
    results["actual_y_perf_of_selected"] = combined_actual_y_for_selection[combined_best_idx].item()
    results["selection_metric_value"] = combined_selection_values[combined_best_idx].item()

    num_train = train_graph_ids.numel()
    num_valid = valid_graph_ids.numel()
    if combined_best_idx < num_train:
        results["selected_graph_origin"] = "Train"
    elif combined_best_idx < num_train + num_valid:
        results["selected_graph_origin"] = "Valid"
    else:
        results["selected_graph_origin"] = "Test"

    rank_info = calculate_overall_rank(
        results["actual_y_perf_of_selected"],
        overall_actual_y,
        higher_is_better
    )
    if rank_info:
        results.update(rank_info)

    return results