import ast
import importlib.util
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rdb2g_precomputed_loader", ROOT / "rdb2g_bench/dataset/dataloader.py"
)
LOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOADER)


class PrecomputedAccessTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        results = Path(scratch.name) / "results"
        csv = results / "tables/rel-f1/driver-top3/hf/GraphSAGE/0.csv"
        csv.parent.mkdir(parents=True)
        csv.write_text(
            "idx,test_metric,gnn,params,train_time\n0,0.75,GraphSAGE,1000,2.5\n",
            encoding="utf-8",
        )
        self.bench = LOADER.RDB2GBench(results)

    def test_gnn_selection_and_values(self):
        self.assertEqual(self.bench.get_available(), {"rel-f1": {"driver-top3": ["GraphSAGE"]}})
        task = self.bench["rel-f1"]["driver-top3"]
        self.assertEqual(task.get_available_gnns(), ["GraphSAGE"])
        with self.assertRaisesRegex(KeyError, "GNN '0' not found"):
            task[0]
        result = task["GraphSAGE"][0]
        self.assertEqual(result["test_metric"], 0.75)
        self.assertEqual(result["params"], 1000)
        self.assertEqual(result["train_time"], 2.5)

    def test_readme_quick_start_uses_working_access_chain(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        section = readme.split("### Download Pre-computed Datasets", 1)[1].split("### ", 1)[0]
        code = re.search(r"```python\n(.*?)```", section, re.DOTALL).group(1)
        names = {"result", "test_metric", "params", "train_time"}
        assignments = [node for node in ast.parse(code).body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id in names]
        self.assertEqual({node.targets[0].id for node in assignments}, names)
        namespace = {"bench": self.bench}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "README.md", "exec"), namespace)
        self.assertEqual(namespace["test_metric"], 0.75)
        self.assertEqual(namespace["params"], 1000)
        self.assertEqual(namespace["train_time"], 2.5)
