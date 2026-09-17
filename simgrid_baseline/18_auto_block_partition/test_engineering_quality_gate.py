"""Tests for the engineering evidence gate."""

import importlib.util
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "engineering_quality_gate", HERE / "engineering_quality_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class EngineeringQualityGateTests(unittest.TestCase):
    def test_current_artifacts_pass_software_gate_but_not_hardware_claim(self):
        result = gate.run_gate()
        self.assertEqual("pass_with_hardware_warnings", result["status"])
        self.assertTrue(result["software_ready_for_hardware_calibration"])
        self.assertFalse(result["target_performance_claim_ready"])
        self.assertEqual(0, result["summary"]["blocks"])
        self.assertGreaterEqual(result["summary"]["warnings"], 1)

    def test_gate_keeps_sensitivity_results_out_of_target_claims(self):
        result = gate.run_gate()
        names = {check["name"]: check for check in result["checks"]}
        self.assertEqual("pass", names[
            "evidence_is_not_labeled_as_hardware_prediction"]["status"])
        self.assertEqual("warn", names[
            "heldout_end_to_end_validation_is_planned_but_not_completed"]["status"])
        self.assertTrue(any("held-out" in reason.lower()
                            for reason in result["why_target_claim_not_ready"]))


if __name__ == "__main__":
    unittest.main()
