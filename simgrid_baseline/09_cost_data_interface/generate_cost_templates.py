import csv
import json
import sys
from pathlib import Path


COST_FIELDS = [
    "task_id",
    "task_type",
    "device",
    "input_shape_batch1",
    "output_shape_batch1",
    "precision",
    "latency_mean_us",
    "latency_p50_us",
    "latency_p95_us",
    "samples",
    "source_kind",
    "source_ref",
    "hardware",
    "notes",
]


def shape_text(shape):
    return "x".join(str(value) for value in shape)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: python {sys.argv[0]} <coarse-dag.json> <output-directory>")

    dag_path = Path(sys.argv[1]).resolve()
    output_directory = Path(sys.argv[2]).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    dag = json.loads(dag_path.read_text(encoding="utf-8"))

    rows = []
    for task in dag["tasks"]:
        for device in task["candidate_devices"]:
            rows.append(
                {
                    "task_id": task["id"],
                    "task_type": task["task_type"],
                    "device": device,
                    "input_shape_batch1": shape_text(task["input_shape_batch1"]),
                    "output_shape_batch1": shape_text(task["output_shape_batch1"]),
                    "precision": "FP32" if device == "GPU" else "",
                    "latency_mean_us": "",
                    "latency_p50_us": "",
                    "latency_p95_us": "",
                    "samples": "",
                    "source_kind": "",
                    "source_ref": "",
                    "hardware": "",
                    "notes": "",
                }
            )

    costs_path = output_directory / "task_costs.csv"
    with costs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    platform_profile = {
        "schema_version": 1,
        "profile_kind": "target_measurement",
        "status": "incomplete",
        "batch": 1,
        "devices": {
            "GPU": {
                "model": None,
                "power_mode": None,
                "software_stack": None,
            },
            "FPGA": {
                "model": None,
                "bitstream_or_design": None,
                "clock_mhz": None,
            },
        },
        "communication": {
            "path": None,
            "GPU_to_FPGA": {
                "fixed_latency_us": None,
                "effective_bandwidth_GBps": None,
                "samples": None,
                "source_kind": None,
                "source_ref": None,
            },
            "FPGA_to_GPU": {
                "fixed_latency_us": None,
                "effective_bandwidth_GBps": None,
                "samples": None,
                "source_kind": None,
                "source_ref": None,
            },
            "supports_bidirectional_overlap": None,
            "supports_compute_transfer_overlap": None,
        },
    }
    profile_path = output_directory / "platform_profile.json"
    profile_path.write_text(json.dumps(platform_profile, indent=2), encoding="utf-8")

    summary = {
        "coarse_tasks": len(dag["tasks"]),
        "cost_rows": len(rows),
        "gpu_rows": sum(row["device"] == "GPU" for row in rows),
        "fpga_rows": sum(row["device"] == "FPGA" for row in rows),
        "task_costs": str(costs_path),
        "platform_profile": str(profile_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
