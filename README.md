# RDB2G-Bench 

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Datasets-blue)](https://huggingface.co/datasets/kaistdata/RDB2G-Bench)

This is an offical implementation of the paper **RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases.**

RDB2G-Bench is a toolkit for benchmarking graph-based analysis and prediction tasks by converting relational database data into graphs.

Our dataset is available at [huggingface ](https://huggingface.co/datasets/kaistdata/RDB2G-Bench).

## Overview

RDB2G-Bench leverages the RelBench datasets to transform relational data into graphs and evaluates the performance of various analysis methods. Key features include:

- Construct Extensive dataset covering 5 real-world RDBs and 12 predictive tasks.
- Performance evaluation of various search methods (Greedy, Evolutionary Algorithm, LLM, etc.).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chlehdwon/RDB2G-Bench.git
cd RDB2G-Bench
```

2. Install the main dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Reproducing our dataset (or you can find [here](https://huggingface.co/datasets/kaistdata/RDB2G-Bench))
#### Key Parameters

- `--dataset`: Name of the RelBench dataset to use (default: "rel-f1")
- `--task`: Name of the task to perform (default: "driver-top3")

#### 1.1 Classification & Regression Task

```bash
python gnn_node_worker.py --dataset [dataset_name] --task [task_name]
```

#### 1.2 Recommendation Task

```bash
python idgnn_link_worker.py --dataset [dataset_name] --task [task_name]
```

### 2. Running the Basic Benchmark

Note: Please verify that all datasets are saved in the result directory specified by `--result_dir` parameter before running the benchmark. (default: `./benchmark/results`)

#### 2.1 Action-based Baselines

```bash
python run_benchmark.py --dataset rel-f1 --task driver-top3 --budget_percentage 0.05 --method all --result_dir [result_dir] 
```

#### Key Parameters

- `--dataset`: Name of the RelBench dataset to use (default: "rel-f1")
- `--task`: Name of the task to perform (default: "driver-top3")
- `--budget_percentage`: Budget Ratio (default: 0.05)
- `--method`: Analysis method ('all', 'ea', 'greedy', 'rl', 'bo')

#### 2.2 LLM-based Baseline

Note: Please replace `"YOUR_API_KEY"` into your private key in `./benchmark/llm/llm_autog.py`.

```bash
python ./benchmark/llm/llm_autog.py --dataset rel-f1 --task driver-top3 --budget_percentage 0.05 --temperature 0.8 --result_dir [result_dir]
```

#### Key Parameters

- `--dataset`: Name of the RelBench dataset to use (default: "rel-f1")
- `--task`: Name of the task to perform (default: "driver-top3")
- `--budget_percentage`: Budget Ratio (default: 0.05)
- `--temperature`: LLM temperature (default: 0.8)

## Reference

The dataset construction and implementation of RDB2G-Bench based on [RelBench](https://github.com/snap-stanford/relbench) framework.


## License

This project is distributed under the MIT License as specified in the LICENSE file.

