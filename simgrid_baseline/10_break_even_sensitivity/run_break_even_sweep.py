import csv
import json
import math
import sys
from pathlib import Path


OUTPUT_FIELDS = [
    "task_type",
    "input_shape_batch1",
    "output_shape_batch1",
    "activation_precision",
    "input_bytes",
    "output_bytes",
    "round_trip_bytes",
    "pcie_effective_bandwidth_GBps",
    "one_way_fixed_latency_us",
    "fpga_speedup_over_gpu",
    "communication_time_us",
    "minimum_gpu_latency_for_offload_us",
]


def element_count(shape):
    return math.prod(shape)


def communication_time_us(input_bytes, output_bytes, bandwidth_gbps, fixed_latency_us):
    transfer_time_us = (input_bytes + output_bytes) / (bandwidth_gbps * 1000.0)
    return 2.0 * fixed_latency_us + transfer_time_us


def minimum_gpu_latency_us(comm_time_us, fpga_speedup):
    if fpga_speedup <= 1:
        raise ValueError("FPGA speedup must be greater than one")
    return comm_time_us / (1.0 - 1.0 / fpga_speedup)


def unique_linear_types(dag):
    unique = {}
    for task in dag["tasks"]:
        if task["candidate_devices"] != ["GPU", "FPGA"]:
            continue
        key = (
            task["task_type"],
            tuple(task["input_shape_batch1"]),
            tuple(task["output_shape_batch1"]),
        )
        unique[key] = unique.get(key, 0) + 1
    return unique


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            f"Usage: python {sys.argv[0]} <coarse-dag.json> <sensitivity-config.json> <output-directory>"
        )

    dag_path = Path(sys.argv[1]).resolve()
    config_path = Path(sys.argv[2]).resolve()
    output_directory = Path(sys.argv[3]).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    dag = json.loads(dag_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config.get("profile_kind") != "sensitivity":
        raise RuntimeError("This script only accepts profile_kind=sensitivity")

    linear_types = unique_linear_types(dag)
    rows = []
    for (task_type, input_shape, output_shape), _count in sorted(linear_types.items()):
        input_elements = element_count(input_shape)
        output_elements = element_count(output_shape)
        for precision, bytes_per_element in config["activation_bytes_per_element"].items():
            input_bytes = input_elements * bytes_per_element
            output_bytes = output_elements * bytes_per_element
            for bandwidth in config["pcie_effective_bandwidth_GBps"]:
                for fixed_latency in config["one_way_fixed_latency_us"]:
                    comm_time = communication_time_us(
                        input_bytes, output_bytes, bandwidth, fixed_latency
                    )
                    for speedup in config["fpga_speedups_over_gpu"]:
                        threshold = minimum_gpu_latency_us(comm_time, speedup)
                        rows.append(
                            {
                                "task_type": task_type,
                                "input_shape_batch1": "x".join(map(str, input_shape)),
                                "output_shape_batch1": "x".join(map(str, output_shape)),
                                "activation_precision": precision,
                                "input_bytes": input_bytes,
                                "output_bytes": output_bytes,
                                "round_trip_bytes": input_bytes + output_bytes,
                                "pcie_effective_bandwidth_GBps": bandwidth,
                                "one_way_fixed_latency_us": fixed_latency,
                                "fpga_speedup_over_gpu": speedup,
                                "communication_time_us": f"{comm_time:.6f}",
                                "minimum_gpu_latency_for_offload_us": f"{threshold:.6f}",
                            }
                        )

    output_path = output_directory / "break_even_sweep.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    representative = [
        row
        for row in rows
        if row["activation_precision"] == "FP32"
        and row["pcie_effective_bandwidth_GBps"] == 4
        and row["one_way_fixed_latency_us"] == 20
        and row["fpga_speedup_over_gpu"] == 4
    ]
    representative.sort(key=lambda row: row["task_type"])
    summary = {
        "profile_kind": "sensitivity",
        "unique_linear_types": len(linear_types),
        "linear_task_instances": sum(linear_types.values()),
        "sweep_rows": len(rows),
        "formula": "T_gpu_min = T_comm / (1 - 1 / speedup)",
        "representative_condition": {
            "activation_precision": "FP32",
            "pcie_effective_bandwidth_GBps": 4,
            "one_way_fixed_latency_us": 20,
            "fpga_speedup_over_gpu": 4,
        },
        "representative_thresholds": [
            {
                "task_type": row["task_type"],
                "communication_time_us": float(row["communication_time_us"]),
                "minimum_gpu_latency_for_offload_us": float(
                    row["minimum_gpu_latency_for_offload_us"]
                ),
            }
            for row in representative
        ],
        "output": str(output_path),
    }
    summary_path = output_directory / "break_even_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
