import csv
import json
import sys
from pathlib import Path


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key):
    return float(row[key])


def select(rows, case, policy, requests):
    matches = [
        row
        for row in rows
        if row["case"] == case
        and row["policy"] == policy
        and int(row["request_count"]) == requests
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row for {case}/{policy}/{requests}, got {len(matches)}")
    return matches[0]


def main():
    results = Path(sys.argv[1]).resolve()
    rows = load_rows(results / "optimization_ablation.csv")
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))

    baseline_gpu = select(rows, "baseline_linear_only", "gpu_only", 1)
    baseline_static = select(rows, "baseline_linear_only", "static_all_fpga", 1)
    baseline_eft = select(rows, "baseline_linear_only", "communication_eft", 1)
    compression_2x = select(rows, "compression_2x", "static_all_fpga", 1)
    compression_4x = select(rows, "compression_4x", "static_all_fpga", 1)
    fused_eft = select(rows, "ffn_fusion_residency", "communication_eft", 1)
    full_single = select(
        rows, "fusion_compression_one_copy_single_buffer", "communication_eft", 1
    )
    full_double = select(
        rows, "fusion_compression_one_copy_double_buffer", "communication_eft", 1
    )
    pipeline_single_buffer = select(
        rows, "fusion_compression_one_copy_single_buffer", "static_all_fpga", 8
    )
    pipeline_double_buffer = select(
        rows, "fusion_compression_one_copy_double_buffer", "static_all_fpga", 8
    )
    shared = select(
        rows, "fusion_compression_one_copy_double_buffer", "communication_eft", 8
    )
    direct = select(
        rows, "fusion_compression_direct_dma_double_buffer", "communication_eft", 8
    )

    checks = {
        "row_count_54": len(rows) == 54,
        "unique_case_policy_request": len(
            {(row["case"], row["policy"], row["request_count"]) for row in rows}
        )
        == len(rows),
        "all_sensitivity_only": all(row["valid_for_target_prediction"] == "False" for row in rows),
        "baseline_gpu_regression": abs(as_float(baseline_gpu, "simgrid_makespan_us") - 22184.375872) < 1e-6,
        "baseline_static_regression": abs(as_float(baseline_static, "simgrid_makespan_us") - 48870.450448) < 1e-6,
        "baseline_eft_regression": abs(as_float(baseline_eft, "simgrid_makespan_us") - 22184.375872) < 1e-6,
        "compression_monotonic_static_latency": as_float(compression_4x, "simgrid_makespan_us")
        < as_float(compression_2x, "simgrid_makespan_us")
        < as_float(baseline_static, "simgrid_makespan_us"),
        "fusion_selects_24_linear_equivalents": int(fused_eft["offloaded_linear_tasks"]) == 24,
        "full_stack_single_request_profitable": as_float(full_single, "latency_speedup_vs_gpu_only") > 1.0,
        "buffer_count_does_not_change_single_request": abs(
            as_float(full_single, "simgrid_makespan_us")
            - as_float(full_double, "simgrid_makespan_us")
        )
        < 1e-6,
        "double_buffer_improves_static_throughput": as_float(
            pipeline_double_buffer, "throughput_requests_per_s"
        )
        > as_float(pipeline_single_buffer, "throughput_requests_per_s"),
        "matched_shared_direct_equivalence": abs(
            as_float(shared, "simgrid_makespan_us") - as_float(direct, "simgrid_makespan_us")
        )
        < 1e-6,
        "fused_dag_shape": summary["fused_dag_tasks"] == 101
        and summary["fused_dag_edges"] == 124,
        "task_and_transfer_records_complete": all(
            int(row["task_records"]) == int(row["tasks_per_request"]) * int(row["request_count"])
            and int(row["transfer_records"])
            == int(row["cross_device_edges"])
            * (2 if row["path_kind"] == "host_staged_two_copy" else 1)
            * int(row["request_count"])
            for row in rows
        ),
    }
    report = {
        "valid": all(checks.values()),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
    }
    (results / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
