# RDB2G-Bench 

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Datasets-blue)](https://huggingface.co/datasets/kaistdata/RDB2G-Bench)

This is an offical implementation of RDB2G-Bench.

RDB2G-Bench is a toolkit for benchmarking graph-based analysis and prediction tasks by converting relational database data into graphs.

Our dataset is available at [huggingface ](https://huggingface.co/datasets/kaistdata/RDB2G-Bench).

## Overview

RDB2G-Bench leverages the RelBench datasets to transform relational data into graphs and evaluates the performance of various analysis methods. Key features include:

- Construct Extensive dataset covering 5 real-world RDBs and 12 predictive tasks.
- Performance evaluation of various search methods (Greedy, Evolutionary Algorithm, LLM, etc.).

## Usage

### Reproduce the RDB2G-Bench dataset (or you can find [here](https://huggingface.co/datasets/kaistdata/RDB2G-Bench))
#### Key Parameters

- `--dataset`: Name of the RelBench dataset to use (default: "rel-f1")
- `--task`: Name of the task to perform (default: "driver-top3")

#### Classification & Regression Task

```bash
python gnn_node_worker.py --dataset [dataset_name] --task [task_name]
```

#### Recommendation Task

```bash
python idgnn_link_worker.py --dataset [dataset_name] --task [task_name]
```

### Running the Basic Benchmark

Note: Please verify that all datasets are saved in the result directory specified by `--result_dir` parameter before running the benchmark.

```bash
python run_benchmark.py --dataset rel-f1 --task driver-top3 --method all --result_dir [result_dir]
```

### Key Parameters

- `--dataset`: Name of the RelBench dataset to use (default: "rel-f1")
- `--task`: Name of the task to perform (default: "driver-top3")
- `--method`: Analysis method ('all', 'gnn', 'ea', 'greedy', 'rl', 'bo')
- `--seed`: Random seed (default: 0)


## Reference

The dataset construction and implementation of RDB2G-Bench based on [RelBench](https://github.com/snap-stanford/relbench) framework.


## License

This project is distributed under the MIT License as specified in the LICENSE file.

