import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validate_gpu_baseline.py")
SPEC = importlib.util.spec_from_file_location("validate_gpu_baseline", MODULE_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


FIELDS = [
    "aux_streams", "cuda_graph", "include_transfers", "precision", "batch",
    "median_ms", "p95_ms", "mean_ms", "throughput_requests_per_s", "enqueue_ms",
    "h2d_ms", "d2h_ms", "samples", "model_sha256", "engine_sha256",
    "jetson_sku", "power_mode", "trt_version", "raw_log",
]


class StrongGpuBaselineTest(unittest.TestCase):
    def write_rows(self, rows):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "measurements.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return temp, path

    def complete_rows(self):
        rows = []
        for aux, graph, transfers in VALIDATOR.REQUIRED_KEYS:
            rows.append({
                "aux_streams": aux, "cuda_graph": str(graph).lower(),
                "include_transfers": str(transfers).lower(), "precision": "FP16", "batch": 1,
                "median_ms": 10 + aux, "p95_ms": 11 + aux, "mean_ms": 10.5 + aux,
                "throughput_requests_per_s": 90, "enqueue_ms": 0.2,
                "h2d_ms": 0.1 if transfers else 0, "d2h_ms": 0.1 if transfers else 0,
                "samples": 500, "model_sha256": "model", "engine_sha256": f"engine-{aux}",
                "jetson_sku": "fixture", "power_mode": "fixture", "trt_version": "fixture",
                "raw_log": "fixture.log",
            })
        return rows

    def test_complete_matrix_is_accepted(self):
        temp, path = self.write_rows(self.complete_rows())
        self.addCleanup(temp.cleanup)
        result = VALIDATOR.validate(path)
        self.assertTrue(result["valid_for_gpu_baseline_claim"])
        self.assertEqual(result["measured_rows"], 16)

    def test_incomplete_matrix_is_not_a_strong_baseline(self):
        temp, path = self.write_rows(self.complete_rows()[:-1])
        self.addCleanup(temp.cleanup)
        result = VALIDATOR.validate(path)
        self.assertFalse(result["valid_for_gpu_baseline_claim"])
        self.assertEqual(len(result["missing_configurations"]), 1)


if __name__ == "__main__":
    unittest.main()
