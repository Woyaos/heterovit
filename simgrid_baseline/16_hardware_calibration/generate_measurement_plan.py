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


def collect_dags(paths):
    dags = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    tasks = {}
    for dag in dags:
        for task in dag["tasks"]:
            prior = tasks.get(task["id"])
            if prior is not None and (prior["task_type"] != task["task_type"] or
                                      prior.get("input_shape_batch1") != task.get("input_shape_batch1") or
                                      prior.get("output_shape_batch1") != task.get("output_shape_batch1")):
                raise ValueError(f"Conflicting task definition for {task['id']}")
            tasks[task["id"]] = task
    return dags, list(tasks.values())


def measurement_devices(task, capability):
    if capability is None:
        return task["candidate_devices"]
    devices = ["GPU"] if "GPU" in task["candidate_devices"] else []
    if "FPGA" in task["candidate_devices"]:
        supported = task["task_type"] in capability["supported_fpga_task_types"]
        if task["task_type"] == "fused_ffn_island":
            types = [item["task_type"] for item in task["fused_components"]]
            supported = types in capability["supported_fpga_blocks"]
        if supported:
            devices.append("FPGA")
    return devices


def shape_text(shape):
    return "x".join(str(value) for value in shape)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def guard_existing_measurements(output):
    for name in ("task_measurements.csv", "transfer_measurements.csv",
                 "heldout_end_to_end_measurements.csv"):
        path = output / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            if any(row.get("latency_us") or row.get("latency_p50_us") or
                   row.get("source_ref") for row in csv.DictReader(handle)):
                raise RuntimeError(f"Refusing to overwrite measured data in {path}")
    platform_path = output / "platform_profile.json"
    if platform_path.exists():
        platform = json.loads(platform_path.read_text(encoding="utf-8"))
        entered = (
            platform.get("activation", {}).get("precision"),
            platform.get("activation", {}).get("bytes_per_element"),
            platform.get("devices", {}).get("GPU", {}).get("model"),
            platform.get("devices", {}).get("FPGA", {}).get("model"),
            platform.get("communication", {}).get("path_kind"),
            platform.get("pipeline", {}).get("fpga_input_buffers"),
            platform.get("measurement_protocol", {}).get("measured_iterations"),
        )
        if any(value is not None for value in entered):
            raise RuntimeError(f"Refusing to overwrite filled platform data in {platform_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dag", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--additional-dag", type=Path, action="append", default=[])
    parser.add_argument("--capability", type=Path)
    parser.add_argument("--activation-bytes-per-element", type=int, default=4)
    args = parser.parse_args()
    if args.activation_bytes_per_element <= 0:
        parser.error("--activation-bytes-per-element must be positive")

    dags, tasks = collect_dags([args.dag, *args.additional_dag])
    dag = dags[0]
    capability = (json.loads(args.capability.read_text(encoding="utf-8"))
                  if args.capability else None)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    guard_existing_measurements(output)

    task_rows = []
    for task in tasks:
        for device in measurement_devices(task, capability):
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

    dag_sizes = {int(round(edge["bytes_batch1_fp32"] *
                           args.activation_bytes_per_element / 4)) for item in dags
                 for edge in item["edges"]}
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

    holdout_fields = ["policy", "sample_id", "latency_us", "source_kind",
                      "source_ref", "dataset_role", "model_variant",
                      "activation_precision"]
    holdout_rows = [{"policy": policy, "sample_id": "", "latency_us": "",
                     "source_kind": "", "source_ref": "", "dataset_role": "",
                     "model_variant": dag.get("metadata", {}).get("model_variant", ""),
                     "activation_precision": ""}
                    for policy in ("gpu_only", "split_ffn_linears", "fixed_ffn",
                                   "automatic_region_dp")]
    write_csv(output / "heldout_end_to_end_measurements.csv",
              holdout_fields, holdout_rows)

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
        "tasks": len(tasks),
        "edges": sum(len(item["edges"]) for item in dags),
        "input_dags": [str(path.resolve()) for path in (args.dag, *args.additional_dag)],
        "capability_status": None if capability is None else capability["status"],
        "activation_bytes_per_element_planned": args.activation_bytes_per_element,
        "task_measurement_rows": len(task_rows),
        "gpu_rows": sum(row["device"] == "GPU" for row in task_rows),
        "fpga_rows": sum(row["device"] == "FPGA" for row in task_rows),
        "transfer_payload_sizes": len(payload_sizes),
        "transfer_measurement_rows": len(transfer_rows),
        "heldout_end_to_end_template_rows": len(holdout_rows),
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
