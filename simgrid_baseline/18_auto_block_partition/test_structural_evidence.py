"""Check that the FFN argument uses actual dependency-edge sizes and costs."""

import copy
import importlib.util
import json
import math
import unittest
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("structural_evidence", HERE / "structural_evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class StructuralEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = evidence.partition.load_runtime()
        cls.dag = json.loads(evidence.partition.DAG_PATH.read_text(encoding="utf-8"))
        cls.profile = json.loads(evidence.partition.PROFILE_PATH.read_text(encoding="utf-8"))
        cls.capability = json.loads(evidence.partition.CAPABILITY_PATH.read_text(encoding="utf-8"))

    def test_actual_ffn_expansion_and_boundary_cost(self):
        model = self.runtime.SensitivityCostModel(self.profile)
        regions, _ = evidence.partition.ffns_from_dependencies(self.dag)
        self.assertEqual(12, len(regions))
        for region in regions:
            row = evidence.ffn_boundary(region, model)
            entry, first, second, output = row["edge_bytes_uncompressed"]
            self.assertEqual(entry, output)
            self.assertEqual(first, second)
            self.assertEqual(4 * entry, first)
            self.assertEqual(2 * entry, row["fused_logical_bytes"])
            self.assertEqual(10 * entry, row["split_logical_bytes"])
            self.assertTrue(math.isclose(0.8, row["logical_byte_reduction_ratio"]))
            expected_saved = (evidence.partition.transfer_us(region["internal_edges"][0],
                                                            "FPGA", "GPU", model)
                              + evidence.partition.transfer_us(region["internal_edges"][1],
                                                               "GPU", "FPGA", model))
            self.assertTrue(math.isclose(expected_saved,
                                         row["saved_internal_transfer_us"], abs_tol=1e-6))

    def test_full_dag_split_and_unavailable_fusion(self):
        result = evidence.run(self.dag, self.profile, self.capability, self.runtime)
        cases = result["full_dag_comparison"]
        self.assertEqual(12, result["selected_full_ffns"])
        self.assertEqual(48, cases["declared_capability_auto"]["physical_transfer_stages"])
        self.assertEqual(96, cases["split_two_linears_fpga_gelu_gpu"]["physical_transfer_stages"])
        self.assertEqual(0, cases["no_full_ffn_kernel_auto"]["cross_device_edges"])
        self.assertEqual(cases["gpu_only"]["makespan_us"],
                         cases["no_full_ffn_kernel_auto"]["makespan_us"])
        self.assertLess(cases["declared_capability_auto"]["makespan_us"],
                        cases["split_two_linears_fpga_gelu_gpu"]["makespan_us"])
        self.assertFalse(result["valid_for_target_prediction"])

    def test_capability_memory_limit_rejects_full_ffn(self):
        capability = copy.deepcopy(self.capability)
        capability["max_workspace_bytes"] = 1
        result = evidence.run(self.dag, self.profile, capability, self.runtime)
        self.assertEqual(0, result["selected_full_ffns"])
        self.assertTrue(all(not row["fused_kernel_supported_by_declared_capability"]
                            for row in result["fusion_condition_by_ffn"]))

    def test_new_simgrid_replays_obey_dag_and_transfer_counts(self):
        for policy, task_count, transfer_count in (
                ("split_ffn_linears", 125, 96), ("fixed_ffn", 101, 48),
                ("linear_only_auto", 125, 0)):
            directory = HERE / "results" / "structural_replay" / policy
            summary = json.loads((directory / "simgrid_replay_summary.json").read_text(
                encoding="utf-8"))
            trace = json.loads((directory / "simgrid_trace.json").read_text(encoding="utf-8"))
            dag = (self.dag if policy != "fixed_ffn" else
                   json.loads((directory / "mixed_dag.json").read_text(encoding="utf-8")))
            self.assertEqual(task_count, summary["task_count"])
            self.assertEqual(transfer_count, summary["transfer_stage_count"])
            self.assertEqual(task_count, len(trace["tasks"]))
            self.assertEqual(transfer_count, len(trace["transfers"]))
            by_id = {row["task_id"]: row for row in trace["tasks"]}
            self.assertEqual(task_count, len(by_id))
            for edge in dag["edges"]:
                self.assertLessEqual(by_id[edge["source"]]["finish_us"],
                                     by_id[edge["target"]]["start_us"] + 1e-5)
            stages = defaultdict(list)
            for row in trace["transfers"]:
                stages[row["edge_index"]].append(row)
            self.assertEqual(transfer_count // 2, len(stages))
            for rows in stages.values():
                rows.sort(key=lambda row: row["stage"])
                self.assertEqual([1, 2], [row["stage"] for row in rows])
                self.assertLessEqual(rows[0]["finish_us"], rows[1]["start_us"] + 1e-5)
            self.assertEqual("simgrid_event_replay_sensitivity_parameters",
                             summary["execution_mode"])
            self.assertFalse(summary["valid_for_target_prediction"])


if __name__ == "__main__":
    unittest.main()
