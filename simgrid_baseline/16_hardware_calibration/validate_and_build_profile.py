import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from generate_measurement_plan import collect_dags, measurement_devices


ALLOWED_PATHS = {"host_staged_two_copy", "shared_mapped_one_copy", "direct_dma_one_copy"}
REQUIRED_PLATFORM_PATHS = [
    "activation.precision",
    "activation.bytes_per_element",
    "measurement_protocol.warmup_iterations",
    "measurement_protocol.measured_iterations",
    "measurement_protocol.synchronization_method",
    "devices.GPU.model",
    "devices.GPU.power_mode",
    "devices.GPU.clock_policy",
    "devices.GPU.software_stack",
    "devices.FPGA.model",
    "devices.FPGA.pcie_interface",
    "devices.FPGA.bitstream_or_design",
    "devices.FPGA.clock_mhz",
    "communication.path_kind",
    "communication.memory_mode",
    "communication.dma_engine",
    "communication.supports_bidirectional_overlap",
    "communication.supports_compute_transfer_overlap",
    "activation_compression.enabled",
    "activation_compression.ratio",
    "activation_compression.encode_fixed_us",
    "activation_compression.decode_fixed_us",
    "activation_compression.encode_bandwidth_GBps",
    "activation_compression.decode_bandwidth_GBps",
    "activation_compression.source_ref",
    "pipeline.fpga_input_buffers",
]


def nested(data, dotted_path):
    current = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def positive(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def nonnegative(value):
    try:
        return math.isfinite(float(value)) and float(value) >= 0
    except (TypeError, ValueError):
        return False


def fit_transfer(rows):
    points = [(float(row["payload_bytes"]), float(row["latency_p50_us"])) for row in rows]
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    intercept = mean_y - slope * mean_x
    if slope <= 0:
        raise ValueError("non-positive latency/byte slope")
    predictions = [intercept + slope * x for x, _ in points]
    ss_res = sum((y - prediction) ** 2 for (_, y), prediction in zip(points, predictions))
    ss_tot = sum((y - mean_y) ** 2 for _, y in points)
    return {
        "fixed_latency_us": max(0.0, intercept),
        "effective_bandwidth_GBps": 1.0 / (slope * 1000.0),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "points": len(points),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dag", type=Path)
    parser.add_argument("task_measurements", type=Path)
    parser.add_argument("transfer_measurements", type=Path)
    parser.add_argument("platform_profile", type=Path)
    parser.add_argument("output_profile", type=Path)
    parser.add_argument("--additional-dag", type=Path, action="append", default=[])
    parser.add_argument("--capability", type=Path)
    args = parser.parse_args()

    _dags, tasks = collect_dags([args.dag, *args.additional_dag])
    capability = (json.loads(args.capability.read_text(encoding="utf-8"))
                  if args.capability else None)
    platform = json.loads(args.platform_profile.read_text(encoding="utf-8"))
    task_rows = list(csv.DictReader(args.task_measurements.open(encoding="utf-8")))
    transfer_rows = list(csv.DictReader(args.transfer_measurements.open(encoding="utf-8")))
    failures = []

    expected = {
        (task["id"], device)
        for task in tasks
        for device in measurement_devices(task, capability)
    }
    complete_task_rows = {}
    for row in task_rows:
        key = (row["task_id"], row["device"])
        valid = (
            key in expected
            and row.get("precision") == nested(platform, "activation.precision")
            and row.get("kernel_or_bitstream")
            and positive(row.get("latency_mean_us"))
            and positive(row.get("latency_p50_us"))
            and positive(row.get("latency_p95_us"))
            and positive(row.get("samples"))
            and int(float(row["samples"])) >= 30
            and row.get("source_kind") == "measured"
            and row.get("source_ref")
            and row.get("hardware")
        )
        if valid:
            complete_task_rows[key] = row
    missing_tasks = sorted(expected - set(complete_task_rows))
    if missing_tasks:
        failures.append(f"{len(missing_tasks)} task/device measurements are incomplete")

    missing_platform = [path for path in REQUIRED_PLATFORM_PATHS if nested(platform, path) is None]
    if missing_platform:
        failures.append(f"{len(missing_platform)} platform fields are incomplete")
    path_kind = nested(platform, "communication.path_kind")
    if path_kind is not None and path_kind not in ALLOWED_PATHS:
        failures.append(f"unsupported communication.path_kind={path_kind!r}")
    if capability is not None:
        if capability["status"] != "verified_target_capability":
            failures.append("FPGA capability has not been verified on the target")
        if capability["activation_precision"] != nested(platform, "activation.precision"):
            failures.append("FPGA capability precision differs from platform precision")

    all_directions = {
        "GPU_to_FPGA",
        "FPGA_to_GPU",
        "GPU_to_HOST",
        "HOST_to_GPU",
        "HOST_to_FPGA",
        "FPGA_to_HOST",
    }
    direction_rows = defaultdict(list)
    for row in transfer_rows:
        valid = (
            row.get("direction") in all_directions
            and positive(row.get("payload_bytes"))
            and positive(row.get("latency_mean_us"))
            and positive(row.get("latency_p50_us"))
            and positive(row.get("latency_p95_us"))
            and positive(row.get("samples"))
            and int(float(row["samples"])) >= 30
            and row.get("source_kind") == "measured"
            and row.get("source_ref")
            and row.get("path_kind") == path_kind
        )
        if valid:
            direction_rows[row["direction"]].append(row)

    if path_kind == "host_staged_two_copy":
        required_directions = ("GPU_to_HOST", "HOST_to_GPU", "HOST_to_FPGA", "FPGA_to_HOST")
    else:
        required_directions = ("GPU_to_FPGA", "FPGA_to_GPU")

    transfer_fits = {}
    for direction in required_directions:
        rows = direction_rows[direction]
        if len({row["payload_bytes"] for row in rows}) < 4:
            failures.append(f"{direction} needs at least four measured payload sizes")
            continue
        try:
            transfer_fits[direction] = fit_transfer(rows)
        except ValueError as error:
            failures.append(f"{direction} fit failed: {error}")

    report = {
        "valid": not failures,
        "input_dags": [str(path.resolve()) for path in (args.dag, *args.additional_dag)],
        "capability_status": None if capability is None else capability["status"],
        "expected_task_device_rows": len(expected),
        "complete_task_device_rows": len(complete_task_rows),
        "missing_task_device_rows": [list(item) for item in missing_tasks[:10]],
        "complete_transfer_rows": sum(len(rows) for rows in direction_rows.values()),
        "missing_platform_fields": missing_platform,
        "transfer_fits": transfer_fits,
        "failures": failures,
    }
    report_path = args.output_profile.with_suffix(".validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        print(json.dumps(report, indent=2))
        raise SystemExit(2)

    compression = platform["activation_compression"]
    task_costs = defaultdict(dict)
    for (task_id, device), row in complete_task_rows.items():
        task_costs[task_id][device] = float(row["latency_p50_us"])

    communication_path = {
        "kind": path_kind,
        "directional_fits": transfer_fits,
        "links_share_both_directions": not platform["communication"][
            "supports_bidirectional_overlap"
        ],
    }
    if path_kind == "host_staged_two_copy":
        gpu_host = [transfer_fits[name] for name in ("GPU_to_HOST", "HOST_to_GPU")]
        host_fpga = [transfer_fits[name] for name in ("HOST_to_FPGA", "FPGA_to_HOST")]
        communication_path["gpu_host"] = {
            "effective_bandwidth_GBps": min(item["effective_bandwidth_GBps"] for item in gpu_host),
            "fixed_latency_us": max(item["fixed_latency_us"] for item in gpu_host),
        }
        communication_path["host_fpga_pcie"] = {
            "effective_bandwidth_GBps": min(
                item["effective_bandwidth_GBps"] for item in host_fpga
            ),
            "fixed_latency_us": max(item["fixed_latency_us"] for item in host_fpga),
        }
    else:
        conservative_fixed = max(fit["fixed_latency_us"] for fit in transfer_fits.values())
        conservative_bandwidth = min(
            fit["effective_bandwidth_GBps"] for fit in transfer_fits.values()
        )
        link_name = (
            "host_fpga_pcie" if path_kind == "shared_mapped_one_copy" else "direct_gpu_fpga"
        )
        communication_path[link_name] = {
            "effective_bandwidth_GBps": conservative_bandwidth,
            "fixed_latency_us": conservative_fixed,
        }

    calibrated = {
        "schema_version": 2,
        "profile_kind": "target_measurement",
        "profile_name": "calibrated_gpu_fpga_target",
        "status": "complete",
        "valid_for_target_prediction": True,
        "activation": platform["activation"],
        "task_costs_us": dict(task_costs),
        "communication_path": communication_path,
        "activation_compression": {
            "ratio": float(compression["ratio"]),
            "encode_fixed_us": float(compression["encode_fixed_us"]),
            "decode_fixed_us": float(compression["decode_fixed_us"]),
            "encode_bandwidth_GBps": float(compression["encode_bandwidth_GBps"]),
            "decode_bandwidth_GBps": float(compression["decode_bandwidth_GBps"]),
        },
        "pipeline": platform["pipeline"],
        "hardware": platform["devices"],
        "measurement_protocol": platform["measurement_protocol"],
        "source_files": {
            "task_measurements": str(args.task_measurements.resolve()),
            "transfer_measurements": str(args.transfer_measurements.resolve()),
            "platform_profile": str(args.platform_profile.resolve()),
            "input_dags": [str(path.resolve()) for path in (args.dag, *args.additional_dag)],
            "capability": None if args.capability is None else str(args.capability.resolve()),
        },
    }
    args.output_profile.write_text(json.dumps(calibrated, indent=2), encoding="utf-8")
    print(json.dumps({"valid": True, "output_profile": str(args.output_profile)}, indent=2))


if __name__ == "__main__":
    main()
