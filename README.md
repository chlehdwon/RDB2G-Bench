# RDB2G-Bench 

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Datasets-blue)](https://huggingface.co/datasets/kaistdata/RDB2G-Bench)
[![arXiv](https://img.shields.io/badge/arXiv-2506.01360-b31b1b.svg)](https://arxiv.org/abs/2506.01360)

This is the official implementation of the paper **RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases.**

**RDB2G-Bench** is a easy-to-use framework for benchmarking graph-based analysis and prediction tasks by converting relational database data into graphs.


## Installation

```bash
git clone https://github.com/chlehdwon/RDB2G-Bench.git
cd RDB2G-Bench
pip install -e .
```

## Python API Usage

### Running Benchmarks

```python
from rdb2g_bench.benchmark.runner import run_benchmark

results = run_benchmark(
    dataset="rel-f1",
    task="driver-top3", 
    budget_percentage=0.05,
    method="all"
)
```

### Running LLM-based baseline

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

### Reproduce Dataset for Classification & Regression Tasks

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

### Reproduce Dataset for Recommendation Tasks

```python
from rdb2g_bench.dataset.link_worker import run_idgnn_link_worker

results = run_idgnn_link_worker(
    dataset="rel-avito",
    task="user-ad-visit",
    gnn_model="GraphSAGE",
    epochs=20,
    lr=0.005
)
```


## Package Structure

```
rdb2g_bench/
├── benchmark/         # Core benchmarking functionality
│   ├── llm/           # LLM-based baseline methods
│   └── baselines/     # Other baseline methods
├── common/            # Shared utilities and search spaces  
├── dataset/           # Dataset loading and processing
└── __init__.py        # Package initialization
```


## Examples

Check the `examples/` directory for complete usage examples and tutorials.


## Reference

The dataset construction and implementation of RDB2G-Bench is based on the [RelBench](https://github.com/snap-stanford/relbench) framework.

## Citation

If you use RDB2G-Bench in your research, please cite:

```bibtex
@article{choi2025rdb2gbench,
    title={RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases}, 
    author={Dongwon Choi and Sunwoo Kim and Juyeon Kim and Kyungho Kim and Geon Lee and Shinhwan Kang and Myunghwan Kim and Kijung Shin},
    year={2025},
    url={https://arxiv.org/abs/2506.01360}, 
}
```

## License

This project is distributed under the MIT License as specified in the LICENSE file.


