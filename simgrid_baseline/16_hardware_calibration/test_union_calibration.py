"""Verify the board measurement plan covers original and fused ViT tasks."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location(
    "measure_plan", HERE / "generate_measurement_plan.py")
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


class UnionCalibrationTests(unittest.TestCase):
    def test_original_and_fused_tasks_have_consistent_measured_candidates(self):
        original = (ROOT / "simgrid_baseline" / "08_coarse_vit_dag" / "results" /
                    "vit_batch1_coarse_dag.json")
        fused = (ROOT / "simgrid_baseline" / "18_auto_block_partition" / "results" /
                 "default" / "mixed_dag.json")
        capability_path = (ROOT / "simgrid_baseline" / "18_auto_block_partition" /
                           "capability_assumptions.json")
        _dags, tasks = plan.collect_dags([original, fused])
        capability = json.loads(capability_path.read_text(encoding="utf-8"))
        pairs = {(task["id"], device) for task in tasks
                 for device in plan.measurement_devices(task, capability)}
        self.assertEqual(137, len(tasks))
        self.assertEqual(173, len(pairs))
        self.assertEqual(36, sum(device == "FPGA" for _, device in pairs))
        self.assertIn(("block_00_ffn_island", "FPGA"), pairs)
        self.assertIn(("block_00_fc1", "GPU"), pairs)
        self.assertNotIn(("block_00_qkv", "FPGA"), pairs)

    def test_linear_only_capability_omits_unavailable_fused_kernel(self):
        fused = (ROOT / "simgrid_baseline" / "18_auto_block_partition" / "results" /
                 "default" / "mixed_dag.json")
        capability_path = (ROOT / "simgrid_baseline" / "18_auto_block_partition" /
                           "capability_linear_only_assumptions.json")
        _dags, tasks = plan.collect_dags([fused])
        capability = json.loads(capability_path.read_text(encoding="utf-8"))
        island = next(task for task in tasks
                      if task["task_type"] == "fused_ffn_island")
        self.assertEqual(["GPU"], plan.measurement_devices(island, capability))

    def test_generator_does_not_replace_entered_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "task_measurements.csv").write_text(
                "task_id,latency_p50_us,source_ref\nlinear,10.0,fixture\n",
                encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                plan.guard_existing_measurements(output)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "platform_profile.json").write_text(json.dumps(
                {"communication": {"path_kind": "host_staged_two_copy"}}),
                encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "filled platform data"):
                plan.guard_existing_measurements(output)


if __name__ == "__main__":
    unittest.main()
