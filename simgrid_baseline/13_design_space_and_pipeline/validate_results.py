import argparse
import csv
import json
from pathlib import Path


POLICIES = {"gpu_only", "static_all_fpga", "communication_eft"}
PATH_STAGES = {
    "host_staged_two_copy": 2,
    "shared_mapped_one_copy": 1,
    "direct_dma_one_copy": 1,
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row, name):
    return float(row[name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    results = args.results_directory.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sweep = read_csv(results / "design_space_sweep.csv")
    replay = read_csv(results / "pipeline_replay.csv")
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))

    conditions = (
        len(config["path_kinds"])
        * len(config["pcie_effective_bandwidth_GBps"])
        * len(config["one_way_fixed_latency_us"])
        * len(config["fpga_linear_speedup_over_gpu"])
    )
    expected_sweep_rows = conditions * len(POLICIES)
    expected_replay_rows = (
        len(config["representative_scenarios"])
        * len(config["request_counts"])
        * len(POLICIES)
    )

    sweep_keys = {
        (
            row["path_kind"],
            row["pcie_effective_bandwidth_GBps"],
            row["one_way_fixed_latency_us"],
            row["fpga_linear_speedup_over_gpu"],
            row["policy"],
        )
        for row in sweep
    }
    gpu_times = {
        key[:-1]: as_float(row, "analytical_makespan_us")
        for row in sweep
        for key in [
            (
                row["path_kind"],
                row["pcie_effective_bandwidth_GBps"],
                row["one_way_fixed_latency_us"],
                row["fpga_linear_speedup_over_gpu"],
                row["policy"],
            )
        ]
        if row["policy"] == "gpu_only"
    }
    eft_not_slower = all(
        as_float(row, "analytical_makespan_us")
        <= gpu_times[
            (
                row["path_kind"],
                row["pcie_effective_bandwidth_GBps"],
                row["one_way_fixed_latency_us"],
                row["fpga_linear_speedup_over_gpu"],
            )
        ]
        + 1e-6
        for row in sweep
        if row["policy"] == "communication_eft"
    )

    one_copy = {}
    one_copy_match = True
    for row in sweep:
        if row["path_kind"] not in {"shared_mapped_one_copy", "direct_dma_one_copy"}:
            continue
        key = (
            row["pcie_effective_bandwidth_GBps"],
            row["one_way_fixed_latency_us"],
            row["fpga_linear_speedup_over_gpu"],
            row["policy"],
        )
        values = (
            row["analytical_makespan_us"],
            row["offloaded_linear_tasks"],
            row["cross_device_edges"],
            row["physical_link_bytes"],
        )
        if key in one_copy and one_copy[key] != values:
            one_copy_match = False
        one_copy[key] = values

    replay_keys = {
        (row["scenario"], row["policy"], row["request_count"]) for row in replay
    }
    replay_records_valid = True
    throughput_values_valid = True
    for row in replay:
        requests = int(row["request_count"])
        if int(row["task_records"]) != 125 * requests:
            replay_records_valid = False
        expected_transfers = (
            int(row["cross_device_edges_per_request"])
            * PATH_STAGES[row["path_kind"]]
            * requests
        )
        if int(row["transfer_records"]) != expected_transfers:
            replay_records_valid = False
        if as_float(row, "throughput_requests_per_s") <= 0:
            throughput_values_valid = False

    checks = {
        "sweep_row_count": len(sweep) == expected_sweep_rows,
        "sweep_keys_unique": len(sweep_keys) == expected_sweep_rows,
        "sweep_policies_complete": {row["policy"] for row in sweep} == POLICIES,
        "sweep_invalid_for_target_prediction": all(
            row["valid_for_target_prediction"] == "False" for row in sweep
        ),
        "communication_eft_never_worse_analytically": eft_not_slower,
        "matched_one_copy_abstractions_are_equal": one_copy_match,
        "replay_row_count": len(replay) == expected_replay_rows,
        "replay_keys_unique": len(replay_keys) == expected_replay_rows,
        "replay_records_complete": replay_records_valid,
        "replay_throughput_positive": throughput_values_valid,
        "replay_invalid_for_target_prediction": all(
            row["valid_for_target_prediction"] == "False" for row in replay
        ),
        "summary_counts_match": (
            summary["hardware_conditions"] == conditions
            and summary["sweep_rows"] == expected_sweep_rows
            and summary["replay_rows"] == expected_replay_rows
        ),
    }
    report = {
        "valid": all(checks.values()),
        "checks": checks,
        "counts": {
            "hardware_conditions": conditions,
            "sweep_rows": len(sweep),
            "replay_rows": len(replay),
        },
    }
    (results / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
