import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    script = Path(__file__).resolve().parent / "validate_and_build_profile.py"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dag = {
            "tasks": [
                {
                    "id": "linear",
                    "task_type": "linear_projection",
                    "candidate_devices": ["GPU", "FPGA"],
                }
            ],
            "edges": [],
        }
        (root / "dag.json").write_text(json.dumps(dag), encoding="utf-8")
        task_fields = [
            "task_id",
            "device",
            "precision",
            "kernel_or_bitstream",
            "latency_mean_us",
            "latency_p50_us",
            "latency_p95_us",
            "samples",
            "source_kind",
            "source_ref",
            "hardware",
        ]
        task_rows = []
        for device, latency in (("GPU", 10.0), ("FPGA", 4.0)):
            task_rows.append(
                {
                    "task_id": "linear",
                    "device": device,
                    "precision": "TEST",
                    "kernel_or_bitstream": "fixture",
                    "latency_mean_us": latency,
                    "latency_p50_us": latency,
                    "latency_p95_us": latency * 1.1,
                    "samples": 30,
                    "source_kind": "measured",
                    "source_ref": "ephemeral-test-fixture",
                    "hardware": "fixture",
                }
            )
        write_csv(root / "tasks.csv", task_fields, task_rows)

        transfer_fields = [
            "direction",
            "payload_bytes",
            "path_kind",
            "latency_mean_us",
            "latency_p50_us",
            "latency_p95_us",
            "samples",
            "source_kind",
            "source_ref",
        ]
        transfer_rows = []
        for direction in ("GPU_to_FPGA", "FPGA_to_GPU"):
            for size in (4096, 16384, 65536, 262144):
                latency = 10.0 + size / 4000.0
                transfer_rows.append(
                    {
                        "direction": direction,
                        "payload_bytes": size,
                        "path_kind": "shared_mapped_one_copy",
                        "latency_mean_us": latency,
                        "latency_p50_us": latency,
                        "latency_p95_us": latency * 1.1,
                        "samples": 30,
                        "source_kind": "measured",
                        "source_ref": "ephemeral-test-fixture",
                    }
                )
        write_csv(root / "transfers.csv", transfer_fields, transfer_rows)

        platform = {
            "activation": {"precision": "TEST", "bytes_per_element": 4},
            "measurement_protocol": {
                "warmup_iterations": 10,
                "measured_iterations": 30,
                "synchronization_method": "fixture",
            },
            "devices": {
                "GPU": {
                    "model": "fixture",
                    "power_mode": "fixture",
                    "clock_policy": "fixture",
                    "software_stack": "fixture",
                },
                "FPGA": {
                    "model": "fixture",
                    "pcie_interface": "fixture",
                    "bitstream_or_design": "fixture",
                    "clock_mhz": 100,
                },
            },
            "communication": {
                "path_kind": "shared_mapped_one_copy",
                "memory_mode": "fixture",
                "dma_engine": "fixture",
                "supports_bidirectional_overlap": False,
                "supports_compute_transfer_overlap": False,
            },
            "activation_compression": {
                "enabled": False,
                "ratio": 1.0,
                "encode_fixed_us": 0.0,
                "decode_fixed_us": 0.0,
                "encode_bandwidth_GBps": 0.0,
                "decode_bandwidth_GBps": 0.0,
                "source_ref": "ephemeral-test-fixture",
            },
            "pipeline": {"fpga_input_buffers": 2},
        }
        (root / "platform.json").write_text(json.dumps(platform), encoding="utf-8")
        output = root / "calibrated.json"
        subprocess.run(
            [
                sys.executable,
                str(script),
                str(root / "dag.json"),
                str(root / "tasks.csv"),
                str(root / "transfers.csv"),
                str(root / "platform.json"),
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        profile = json.loads(output.read_text(encoding="utf-8"))
        link = profile["communication_path"]["host_fpga_pcie"]
        assert abs(link["fixed_latency_us"] - 10.0) < 1e-9
        assert abs(link["effective_bandwidth_GBps"] - 4.0) < 1e-9
        assert profile["task_costs_us"]["linear"] == {"GPU": 10.0, "FPGA": 4.0}
        task_rows[0]["precision"] = "WRONG_PRECISION"
        write_csv(root / "tasks.csv", task_fields, task_rows)
        rejected = subprocess.run(
            [sys.executable, str(script), str(root / "dag.json"),
             str(root / "tasks.csv"), str(root / "transfers.csv"),
             str(root / "platform.json"), str(output)],
            capture_output=True, text=True,
        )
        assert rejected.returncode == 2
        report = json.loads(output.with_suffix(".validation.json").read_text(
            encoding="utf-8"))
        assert ["linear", "GPU"] in report["missing_task_device_rows"]
    print("calibrated-profile builder: PASS")


if __name__ == "__main__":
    main()
