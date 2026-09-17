"""Integrity checks for stored SimGrid traces and evidence labels."""

import csv
import json
import math
import unittest
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class EvidenceTests(unittest.TestCase):
    def test_iris_planned_manifest_matches_dp_placement(self):
        directory = HERE / "results"
        manifest = json.loads((directory / "iris_planned_manifest.json").read_text(encoding="utf-8"))
        placements = json.loads((directory / "default" / "placements.json").read_text(encoding="utf-8"))
        dag = json.loads((directory / "default" / "mixed_dag.json").read_text(encoding="utf-8"))
        header = (directory / "iris_planned_manifest.h").read_text(encoding="ascii")
        self.assertEqual(len(dag["tasks"]), manifest["dag_task_count"])
        self.assertEqual(len(dag["edges"]), manifest["dag_edge_count"])
        self.assertIn("int metadata[9]", header)
        self.assertEqual(set(placements), {task["id"] for task in manifest["tasks"]})
        for task in manifest["tasks"]:
            self.assertEqual(placements[task["id"]], task["planned_device"])
            self.assertEqual(9, len(task["metadata"]))
            self.assertEqual(1 if placements[task["id"]] == "GPU" else 2,
                             task["metadata"][8])
        self.assertEqual(12, sum(task["planned_device"] == "FPGA" for task in manifest["tasks"]))

    def test_batch_replay_has_one_task_per_request_and_valid_dependencies(self):
        matrix = load_csv(HERE / "results" / "replay" / "matrix.csv")
        self.assertEqual(12, len(matrix))
        for item in matrix:
            policy = item["policy"]
            requests = int(item["requests"])
            buffers = int(item["buffers"])
            directory = HERE / "results" / "replay" / policy / f"requests_{requests}_buffers_{buffers}"
            summary = json.loads((directory / "simgrid_replay_summary.json").read_text(encoding="utf-8"))
            trace = json.loads((directory / "simgrid_trace.json").read_text(encoding="utf-8"))
            dag_path = (ROOT / "simgrid_baseline" / "08_coarse_vit_dag" / "results" /
                        "vit_batch1_coarse_dag.json" if policy != "automatic_region_dp" else
                        directory / "mixed_dag.json")
            dag = json.loads(dag_path.read_text(encoding="utf-8"))
            task_count = len(dag["tasks"])
            task_rows = trace["tasks"]
            transfer_rows = trace["transfers"]
            self.assertEqual(requests * task_count, len(task_rows))
            self.assertEqual(requests, len(summary["request_completion_us"]))
            by_request = defaultdict(dict)
            for row in task_rows:
                request_id = row["request_id"]
                task_id = row["task_id"]
                self.assertNotIn(task_id, by_request[request_id])
                by_request[request_id][task_id] = row
            for request_id in range(requests):
                self.assertEqual(task_count, len(by_request[request_id]))
                for edge in dag["edges"]:
                    source = by_request[request_id][edge["source"]]
                    target = by_request[request_id][edge["target"]]
                    self.assertLessEqual(source["finish_us"], target["start_us"] + 1e-5)
            by_device = defaultdict(list)
            for row in task_rows:
                by_device[row["device"]].append(row)
            for rows in by_device.values():
                rows.sort(key=lambda row: row["start_us"])
                for previous, current in zip(rows, rows[1:]):
                    self.assertLessEqual(previous["finish_us"], current["start_us"] + 1e-5)
            stages = defaultdict(list)
            for row in transfer_rows:
                stages[(row["request_id"], row["edge_index"])].append(row)
            for rows in stages.values():
                rows.sort(key=lambda row: row["stage"])
                self.assertEqual([1, 2], [row["stage"] for row in rows])
                self.assertLessEqual(rows[0]["finish_us"], rows[1]["start_us"] + 1e-5)
            self.assertFalse(summary["valid_for_target_prediction"])
            self.assertEqual("simgrid_event_replay_sensitivity_parameters", summary["execution_mode"])

    def test_online_p95_and_completion_accounting(self):
        matrix = load_csv(HERE / "results" / "online" / "online_matrix.csv")
        self.assertEqual(9, len(matrix))
        for item in matrix:
            directory = (HERE / "results" / "online" /
                         f"rate_{item['offered_rate_requests_per_s']}" /
                         f"{item['policy']}_buffers_{item['buffers']}")
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            completions = load_csv(directory / "request_completion.csv")
            tasks = load_csv(directory / "task_timeline.csv")
            transfers = load_csv(directory / "transfer_timeline.csv")
            requests = int(item["requests"])
            task_count = summary["task_records"] // requests
            self.assertEqual(requests, len(completions))
            self.assertEqual(summary["task_records"], len(tasks))
            self.assertEqual(summary["transfer_records"], len(transfers))
            self.assertEqual(Counter({str(i): task_count for i in range(requests)}),
                             Counter(row["request_id"] for row in tasks))
            latencies = sorted(float(row["latency_us"]) for row in completions)
            rank = max(0, math.ceil(0.95 * len(latencies)) - 1)
            self.assertTrue(math.isclose(summary["p95_latency_us"], latencies[rank], abs_tol=1e-5))
            self.assertTrue(math.isclose(float(item["p95_latency_ms"]),
                                          summary["p95_latency_us"] / 1000.0, abs_tol=1e-6))
            self.assertFalse(summary["valid_for_target_prediction"])

    def test_old_rf880_buffer_ablation_is_only_analytical_fallback(self):
        old = (ROOT / "simgrid_baseline" / "17_rf880_agx_final" / "results" /
               "optimization_ablation" / "optimization_ablation.csv")
        rows = load_csv(old)
        by_key = {(row["case"], row["policy"], row["request_count"]): row for row in rows}
        single = "rf880_fusion_compression_shared_single_buffer"
        double = "rf880_fusion_compression_shared_double_buffer"
        comparisons = 0
        for (case, policy, requests), row in by_key.items():
            if case != single:
                continue
            paired = by_key[(double, policy, requests)]
            self.assertEqual("analytical_fallback_no_simgrid", row["execution_mode"])
            self.assertEqual("analytical_fallback_no_simgrid", paired["execution_mode"])
            self.assertEqual(row["simgrid_makespan_us"], paired["simgrid_makespan_us"])
            comparisons += 1
        self.assertGreater(comparisons, 0)


if __name__ == "__main__":
    unittest.main()
