"""Synthetic-only checks for the board-validation analysis interface."""

import importlib.util
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "heldout_latency", HERE / "validate_heldout_latency.py")
heldout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heldout)


def fixture_prediction(policy="gpu_only"):
    return {"policy": policy, "requests": 1, "makespan_us": 1000.0,
            "valid_for_target_prediction": True,
            "execution_mode": "simgrid_event_replay_target_measurement_profile",
            "model_variant": "fixture_vit", "activation_precision": "FP16",
            "profile_name": "synthetic_fixture_profile"}


def fixture_rows(policy="gpu_only"):
    return [{"policy": policy, "sample_id": str(index),
             "latency_us": str(990 + index), "source_kind": "measured",
             "source_ref": "synthetic_test_fixture_not_board_data",
             "dataset_role": "held_out_validation",
             "model_variant": "fixture_vit", "activation_precision": "FP16"}
            for index in range(30)]


class HeldoutLatencyTests(unittest.TestCase):
    def test_reports_median_p95_and_prediction_error(self):
        result = heldout.compare({"gpu_only": fixture_prediction()}, fixture_rows())
        row = result["policies"]["gpu_only"]
        self.assertEqual(30, row["samples"])
        self.assertEqual(1004.5, row["hardware_median_us"])
        self.assertEqual(1018.0, row["hardware_p95_us"])
        self.assertAlmostEqual(4.5 / 1004.5, row["median_absolute_percentage_error"])

    def test_rejects_sensitivity_and_nonheldout_input(self):
        prediction = fixture_prediction()
        prediction["valid_for_target_prediction"] = False
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            heldout.compare({"gpu_only": prediction}, fixture_rows())
        rows = fixture_rows()
        rows[0]["dataset_role"] = "cost_calibration"
        with self.assertRaisesRegex(ValueError, "not a held-out"):
            heldout.compare({"gpu_only": fixture_prediction()}, rows)

    def test_rejects_mismatched_precision_and_duplicate_samples(self):
        rows = fixture_rows()
        rows[0]["activation_precision"] = "FP32"
        with self.assertRaisesRegex(ValueError, "precision differs"):
            heldout.compare({"gpu_only": fixture_prediction()}, rows)
        rows = fixture_rows()
        rows[1]["sample_id"] = rows[0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "duplicate sample ID"):
            heldout.compare({"gpu_only": fixture_prediction()}, rows)

    def test_rejects_predictions_from_different_profiles(self):
        other = fixture_prediction("fixed_ffn")
        other["profile_name"] = "other_target_profile"
        with self.assertRaisesRegex(ValueError, "one named calibrated"):
            heldout.compare({"gpu_only": fixture_prediction(), "fixed_ffn": other},
                            fixture_rows() + fixture_rows("fixed_ffn"))


if __name__ == "__main__":
    unittest.main()
