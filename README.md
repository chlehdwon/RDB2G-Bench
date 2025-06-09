# RDB2G-Bench 

[![Latest Release](https://img.shields.io/badge/Latest-v0.1-success)](https://github.com/chlehdwon/RDB2G-Bench/releases)
[![Read the Docs](https://img.shields.io/readthedocs/RDB2G-Bench)](https://rdb2g-bench.readthedocs.io/en/latest/)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Datasets-blue)](https://huggingface.co/datasets/kaistdata/RDB2G-Bench)
[![arXiv](https://img.shields.io/badge/arXiv-2506.01360-b31b1b.svg)](https://arxiv.org/abs/2506.01360)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official implementation of the paper **RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases.**

**RDB2G-Bench** is an **easy-to-use framework** for benchmarking graph-based analysis and prediction tasks by converting relational database data into graphs.

## 🚀 Installation

```bash
git clone https://github.com/chlehdwon/RDB2G-Bench.git
cd RDB2G-Bench
pip install -e .
```

Also, please install additional PyG dependencies. The below shows an example when you use torch 2.1.0 + cuda 12.1.

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```
You can skip this part if you don't want to reproduce our dataset.

## ⚡ Quick Start

Comprehensive documentation and detailed guides are available at [our documentation site](https://rdb2g-bench.readthedocs.io/en/latest/).

You can also check the `examples/` directory for complete usage examples and tutorials.

### Download Pre-computed Datasets

```python
from rdb2g_bench.dataset.dataset import load_rdb2g_bench

bench = load_rdb2g_bench("./results")

result = bench['rel-f1']['driver-top3'][0]  # Access by graph index
test_metric = result['test_metric']         # Test performance
params = result['params']                   # Model parameters
train_time = result['train_time']           # Train time
```

### Reproduce Datasets for Classification & Regression Tasks

```python
from rdb2g_bench.dataset.node_worker import run_gnn_node_worker

results = run_gnn_node_worker(
    dataset="rel-f1",
    task="driver-top3",
    gnn_model="GraphSAGE",
    epochs=20,
    lr=0.005
)
```

### Reproduce Datasets for Recommendation Tasks

```python
from rdb2g_bench.dataset.link_worker import run_idgnn_link_worker

results = run_idgnn_link_worker(
    dataset="rel-avito",
    task="user-ad-visit",
    gnn_model="GraphSAGE",
    epochs=20,
    lr=0.001
)
```

### Run Benchmarks

```python
from rdb2g_bench.benchmark.runner import run_benchmark

results = run_benchmark(
    dataset="rel-f1",
    task="driver-top3", 
    budget_percentage=0.05,
    method="all",
    num_runs=10,
    seed=0
)
```

### Run LLM-based baseline

Before using LLM-based baseline, you need to set up your API key:

```bash
export ANTHROPIC_API_KEY="YOUR_API_KEY"
```

```python
from rdb2g_bench.benchmark.llm.llm_runner import run_llm_baseline

results = run_llm_baseline(
    dataset="rel-f1",
    task="driver-top3",
    budget_percentage=0.05,
    model="claude-3-5-sonnet-latest",
    temperature=0.8,
    seed=42
)
```

## 📁 Package Structure

```
rdb2g_bench/
├── benchmark/         # Core benchmarking functionality
│   ├── llm/           # LLM-based baseline methods
│   └── baselines/     # Other baseline methods
├── common/            # Shared utilities and search spaces  
├── dataset/           # Dataset loading and processing
└── __init__.py        # Package initialization
```

## 📖 Reference

The dataset construction and implementation of RDB2G-Bench is based on the [RelBench](https://github.com/snap-stanford/relbench) framework.

## 📝 Citation

If you use RDB2G-Bench in your research, please cite:

```bibtex
@article{choi2025rdb2gbench,
    title={RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases}, 
    author={Dongwon Choi and Sunwoo Kim and Juyeon Kim and Kyungho Kim and Geon Lee and Shinhwan Kang and Myunghwan Kim and Kijung Shin},
    year={2025},
    url={https://arxiv.org/abs/2506.01360}, 
}
```

## ⚖️ License

This project is distributed under the MIT License as specified in the LICENSE file.


