import argparse
import json
import os
# os.environ['XDG_CACHE_HOME'] = '/data/cache'
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig

from relbench.base import Dataset, EntityTask, RecommendationTask
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from search_space.gnn_search_space import GNNNodeSearchSpace, GNNLinkSearchSpace, IDGNNLinkSearchSpace
from search_space.search_space import TotalSearchSpace

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patches as patches

class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
parser.add_argument("--task_type", type=str, default="gnn_node")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument(
    "--result_dir",
    type=str,
    required=True,
)
args, unknown = parser.parse_known_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)

if args.task_type == "gnn_node":
    if isinstance(task, RecommendationTask):
        raise ValueError("Task type must not be 'gnn_node' for recommendation tasks")
    search_space_class, src_entity_table, dst_entity_table = GNNNodeSearchSpace, task.entity_table, None
    fig_dir = f"{args.result_dir}/figs/{args.dataset}/{args.task}"
elif args.task_type == "gnn_link":
    if isinstance(task, EntityTask):
        raise ValueError("Task type must be 'gnn_node' for entity tasks")
    search_space_class, src_entity_table, dst_entity_table = GNNLinkSearchSpace, task.src_entity_table, task.dst_entity_table
    fig_dir = f"{args.result_dir}/figs/{args.dataset}/{args.task}/{args.task_type}"
elif args.task_type == "idgnn_link":
    if isinstance(task, EntityTask):
        raise ValueError("Task type must not be 'gnn_node' for recommendation tasks")
    search_space_class, src_entity_table, dst_entity_table = IDGNNLinkSearchSpace, task.src_entity_table, task.dst_entity_table
    fig_dir = f"{args.result_dir}/figs/{args.dataset}/{args.task}/{args.task_type}"
else:
    raise ValueError(f"Unknown task_type: {args.task_type}")

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

search_space = TotalSearchSpace(
    dataset=args.dataset,
    task=args.task,
    hetero_data=data,
    GNNSearchSpace=search_space_class,
    num_layers=args.num_layers,
    src_entity_table=src_entity_table,
    dst_entity_table=dst_entity_table
)

all_graphs = search_space.generate_all_graphs()
full_idx = search_space.get_full_graph_idx(all_graphs)

def create_nx_multidigraph(
    edge_types: List[Tuple[str, str, str]],
    selected_edges_mask: Optional[List[int]] = None
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    all_nodes = set()
    edges_to_add = []

    for i, (src, rel, dst) in enumerate(edge_types):
        all_nodes.add(src)
        all_nodes.add(dst)
        if selected_edges_mask is None or (i < len(selected_edges_mask) and selected_edges_mask[i] == 1):
            edges_to_add.append((src, dst, {"type": rel}))

    for node in all_nodes:
        G.add_node(node, type=node)

    for u, v, data in edges_to_add:
        G.add_edge(u, v, **data)

    return G

# --- Setup Plotting Configuration (Colors, Layout) ---
full_edge_tuples = search_space.get_full_edges()
node_types = list(search_space.row2graph.node_types)
node_cmap = plt.get_cmap("tab10", len(node_types))
node_color_map = {ntype: node_cmap(i) for i, ntype in enumerate(node_types)}
edge_color_map = {"f2p": "black"}
r2e_cmap = plt.get_cmap("tab10", len(node_types))

base_layout_graph = create_nx_multidigraph(full_edge_tuples)
pos = nx.circular_layout(base_layout_graph)

# --- Ensure Output Directory Exists ---
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# --- Iterate Through Each Graph Candidate and Plot ---
for i, selected_mask in enumerate(all_graphs):
    G = create_nx_multidigraph(full_edge_tuples, selected_mask)

    # Remove isolated nodes for cleaner plots
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    if not G.nodes():
        continue

    plt.figure(figsize=(12, 10))

    # --- Draw Nodes and Node Labels ---
    current_nodes = list(G.nodes())
    node_colors = [node_color_map.get(G.nodes[n]['type'], '#DDDDDD') for n in current_nodes]
    labels = {node: node for node in current_nodes}

    if current_nodes:
        x_coords = [pos[n][0] for n in current_nodes if n in pos]
        y_coords = [pos[n][1] for n in current_nodes if n in pos]
        if x_coords and y_coords:
             x_min, x_max = min(x_coords), max(x_coords)
             y_min, y_max = min(y_coords), max(y_coords)
             padding = 0.2
             plt.xlim(x_min - padding, x_max + padding)
             plt.ylim(y_min - padding, y_max + padding)

    nx.draw_networkx_nodes(G, pos, nodelist=current_nodes, node_color=node_colors, node_size=8000, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=14)

    ax = plt.gca()
    r2e_legend_info = {} # Store {target_node_type: color} for legend

    # --- Pre-calculate edge indices for applying curvature to multi-edges ---
    edge_index_map = {}
    num_edges_map = {}
    for u, v, k in G.edges(keys=True):
        num_edges = G.number_of_edges(u, v)
        if num_edges > 1:
            pair_key = (u, v)
            num_edges_map[pair_key] = num_edges
            if pair_key not in edge_index_map:
                edge_index_map[pair_key] = {}
                current_index = 0
                for edge_k in G.get_edge_data(u, v).keys():
                    edge_index_map[pair_key][edge_k] = current_index
                    current_index += 1

    # --- Iterate Through Edges to Draw and Collect Legend Info ---
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_type_str = data.get("type", "")
        arrowstyle = "->"
        edge_color = edge_color_map["f2p"]
        label_text = ""
        is_f2p = False

        # Determine edge properties based on type string
        if edge_type_str.startswith("f2p_"):
            is_f2p = True
            parts = edge_type_str.split("f2p_", 1)
            if len(parts) > 1:
                label_text = parts[1]

        elif edge_type_str.startswith("r2e_"):
            target_node_type = edge_type_str.split("_", 1)[1]
            arrowstyle = "<->"
            try:
                type_index = node_types.index(target_node_type)
                edge_color = r2e_cmap(type_index)
                if target_node_type not in r2e_legend_info:
                    r2e_legend_info[target_node_type] = edge_color
            except ValueError:
                print(f"Warning: Node type '{target_node_type}' from edge '{edge_type_str}' not found. Using default color.")
                edge_color = "grey"

        # Draw edges or self-loop labels
        if u == v: # Handle self-loops (Label only)
            if u not in pos:
                print(f"Warning: Position for self-loop node '{u}' not found. Skipping.")
                continue
            node_pos = pos[u]
            label_offset_factor = 0.15
            # Determine label text for self-loop (can be f2p or r2e origin)
            self_loop_label_text = ""
            if edge_type_str.startswith("f2p_"):
                 parts = edge_type_str.split("f2p_", 1)
                 if len(parts) > 1: self_loop_label_text = parts[1]
            elif edge_type_str.startswith("r2e_"):
                 parts = edge_type_str.split("r2e_", 1)
                 if len(parts) > 1: self_loop_label_text = parts[1]

            if self_loop_label_text:
                label_pos_x = node_pos[0]
                label_pos_y = node_pos[1] + label_offset_factor
                ax.text(label_pos_x, label_pos_y, self_loop_label_text,
                        size=8, color='#333333', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

        else: # Handle non-self-loops (Draw edge and potentially f2p label)
            # Calculate curvature for multi-edges
            rad = 0.0
            pair_key = (u, v)
            if pair_key in num_edges_map and num_edges_map[pair_key] > 1:
                 if key in edge_index_map.get(pair_key, {}):
                     edge_index = edge_index_map[pair_key][key]
                     sign = 1 if edge_index % 2 == 0 else -1
                     magnitude = 0.1 * ( (edge_index // 2) + 1)
                     rad = sign * magnitude

            # Draw the edge
            ax.annotate("",
                        xy=pos[v], xycoords='data',
                        xytext=pos[u], textcoords='data',
                        arrowprops=dict(arrowstyle=arrowstyle, color=edge_color,
                                        shrinkA=50, shrinkB=50,
                                        patchA=None, patchB=None,
                                        connectionstyle=f"arc3,rad={rad}",
                                        linewidth=1.5))

            # Add label for f2p edges, offset if curved
            if is_f2p and label_text:
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                text_x, text_y = mid_x, mid_y

                if rad != 0:
                    dx = pos[v][0] - pos[u][0]
                    dy = pos[v][1] - pos[u][1]
                    norm = (dx**2 + dy**2)**0.5
                    if norm > 0:
                        norm_x = -dy / norm
                        norm_y = dx / norm
                        fixed_label_offset = 0.08
                        text_x = mid_x + norm_x * fixed_label_offset * np.sign(rad)
                        text_y = mid_y + norm_y * fixed_label_offset * np.sign(rad)

                ax.text(text_x, text_y, label_text,
                        size=8, color='#333333',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

    # --- Create and Add Legend for Edge Types ---
    legend_elements = []
    if r2e_legend_info:
        for target_node_type, color in r2e_legend_info.items():
            legend_elements.append(Line2D([0], [0], marker='o',
                                         markerfacecolor=color, markersize=0,
                                         linestyle='-', linewidth=1.5,
                                         color=color,
                                         label=f'{target_node_type}'))

        ax.legend(handles=legend_elements, loc='best', fontsize='small')

    # --- Finalize and Save Plot ---
    plt.axis('off')
    save_path = f"{fig_dir}/graph_{i}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()