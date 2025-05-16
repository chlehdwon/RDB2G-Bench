import argparse
import os

from benchmark.benchmark import main

parser = argparse.ArgumentParser(description='GNN for Performance Prediction')

# Dataset Args
parser.add_argument("--dataset", type=str, default="rel-f1", help="Name of the RelBench dataset")
parser.add_argument("--task", type=str, default="driver-top3", help="Name of the RelBench task")
parser.add_argument("--tag", type=str, default="final", help="Tag identifying the results sub-directory")
parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/relbench_examples"), help="Directory for caching")
parser.add_argument("--result_dir", type=str, default=os.path.expanduser("./results"), help="Root directory for results")

# Model Hyperparameters
parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of node embeddings")
parser.add_argument("--gnn_hidden_dim", type=int, default=64, help="Hidden dimension of GNN layers")
parser.add_argument("--num_gnn_layers", type=int, default=1, help="Number of GNN layers")
parser.add_argument("--mlp_hidden_dim", type=int, default=32, help="Hidden dimension of MLP predictor")
parser.add_argument("--dropout_rate", type=float, default=0, help="Dropout rate")
parser.add_argument("--valid_ratio", type=float, default=0.5, help="Proportion of validation set")

# Evolutionary Heuristic Hyperparameters
parser.add_argument('--evo_pop_size', type=int, default=10, help='Population size for evolutionary heuristic')
parser.add_argument('--evo_tourn_size', type=int, default=10, help='Tournament size for evolutionary heuristic')
parser.add_argument('--evo_max_iter', type=int, default=1000, help='Max iterations for evolutionary heuristic')

# Training Hyperparameters
parser.add_argument("--train_ratio", type=float, default=0.05, help="Combined proportion for train and validation sets (e.g., 0.1 means 5% train, 5% valid).")
parser.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularization)")
parser.add_argument("--seed", type=int, default=0, help="Initial random seed (seed for run i will be seed + i)")
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (if available), -1 for CPU")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
parser.add_argument("--patience", type=int, default=20, help="Number of epochs for early stopping")
parser.add_argument("--num_runs", type=int, default=10, help="Number of times to repeat the training and evaluation process")

parser.add_argument(
    '--method',
    type=str,
    nargs='+',
    default=['all'], 
    choices=['all', 'gnn', 'ea', 'greedy', 'rl', 'bo'],
    help='Which analysis method(s) to run. Specify one or more methods separated by spaces. \'all\' runs all methods.'
)

args = parser.parse_args()
main(args) 