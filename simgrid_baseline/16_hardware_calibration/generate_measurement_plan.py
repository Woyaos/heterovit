import argparse
import csv
import json
import math
from pathlib import Path


TASK_FIELDS = [
    "task_id",
    "task_type",
    "device",
    "input_shape_batch1",
    "output_shape_batch1",
    "precision",
    "kernel_or_bitstream",
    "latency_mean_us",
    "latency_p50_us",
    "latency_p95_us",
    "samples",
    "source_kind",
    "source_ref",
    "hardware",
    "notes",
]
TRANSFER_FIELDS = [
    "direction",
    "payload_bytes",
    "path_kind",
    "latency_mean_us",
    "latency_p50_us",
    "latency_p95_us",
    "samples",
    "source_kind",
    "source_ref",
    "notes",
]


def shape_text(shape):
    return "x".join(str(value) for value in shape)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dag", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    dag = json.loads(args.dag.read_text(encoding="utf-8"))
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)

    task_rows = []
    for task in dag["tasks"]:
        for device in task["candidate_devices"]:
            task_rows.append(
                {
                    "task_id": task["id"],
                    "task_type": task["task_type"],
                    "device": device,
                    "input_shape_batch1": shape_text(task["input_shape_batch1"]),
                    "output_shape_batch1": shape_text(task["output_shape_batch1"]),
                    "precision": "",
                    "kernel_or_bitstream": "",
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
    write_csv(output / "task_measurements.csv", TASK_FIELDS, task_rows)

    dag_sizes = {int(edge["bytes_batch1_fp32"]) for edge in dag["edges"]}
    standard_sizes = {4096, 16384, 65536, 262144, 1048576, 4194304}
    payload_sizes = sorted(dag_sizes | standard_sizes)
    transfer_rows = []
    directions = (
        "GPU_to_FPGA",
        "FPGA_to_GPU",
        "GPU_to_HOST",
        "HOST_to_GPU",
        "HOST_to_FPGA",
        "FPGA_to_HOST",
    )
    for direction in directions:
        for payload_bytes in payload_sizes:
            transfer_rows.append(
                {
                    "direction": direction,
                    "payload_bytes": payload_bytes,
                    "path_kind": "",
                    "latency_mean_us": "",
                    "latency_p50_us": "",
                    "latency_p95_us": "",
                    "samples": "",
                    "source_kind": "",
                    "source_ref": "",
                    "notes": "",
                }
            )
    write_csv(output / "transfer_measurements.csv", TRANSFER_FIELDS, transfer_rows)

    platform = {
        "schema_version": 2,
        "profile_kind": "target_measurement_input",
        "status": "incomplete",
        "model": dag.get("metadata", {}).get("model_variant"),
        "batch": dag.get("metadata", {}).get("batch", 1),
        "activation": {"precision": None, "bytes_per_element": None},
        "measurement_protocol": {
            "warmup_iterations": None,
            "measured_iterations": None,
            "synchronization_method": None,
            "background_load": "idle",
        },
        "devices": {
            "GPU": {
                "model": None,
                "power_mode": None,
                "clock_policy": None,
                "software_stack": None,
            },
            "FPGA": {
                "model": None,
                "pcie_interface": None,
                "bitstream_or_design": None,
                "clock_mhz": None,
            },
        },
        "communication": {
            "path_kind": None,
            "memory_mode": None,
            "dma_engine": None,
            "supports_bidirectional_overlap": None,
            "supports_compute_transfer_overlap": None,
        },
        "activation_compression": {
            "enabled": None,
            "ratio": None,
            "encode_fixed_us": None,
            "decode_fixed_us": None,
            "encode_bandwidth_GBps": None,
            "decode_bandwidth_GBps": None,
            "source_ref": None,
        },
        "pipeline": {"fpga_input_buffers": None},
    }
    (output / "platform_profile.json").write_text(
        json.dumps(platform, indent=2), encoding="utf-8"
    )

    summary = {
        "tasks": len(dag["tasks"]),
        "edges": len(dag["edges"]),
        "task_measurement_rows": len(task_rows),
        "gpu_rows": sum(row["device"] == "GPU" for row in task_rows),
        "fpga_rows": sum(row["device"] == "FPGA" for row in task_rows),
        "transfer_payload_sizes": len(payload_sizes),
        "transfer_measurement_rows": len(transfer_rows),
        "transfer_directions": len(directions),
        "largest_payload_bytes": max(payload_sizes),
        "status": "waiting_for_target_hardware",
    }
    (output / "measurement_plan_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
