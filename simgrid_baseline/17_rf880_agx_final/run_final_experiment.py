import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def run_command(command):
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(str(part) for part in command)
            + "\nSTDOUT:\n"
            + completed.stdout
            + "\nSTDERR:\n"
            + completed.stderr
        )
    return completed


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key):
    return float(row[key])


def as_int(row, key):
    return int(float(row[key]))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simgrid_available():
    try:
        import simgrid  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def placement_metrics(runtime, dag, cost_model, placements):
    cross_edges = [
        edge
        for edge in dag["edges"]
        if placements[edge["source"]] != placements[edge["target"]]
    ]
    logical_bytes = sum(cost_model.edge_bytes(edge) for edge in cross_edges)
    uncompressed_bytes = sum(cost_model.uncompressed_edge_bytes(edge) for edge in cross_edges)
    offloaded = sum(
        runtime.SensitivityCostModel.linear_equivalent_count(task)
        for task in dag["tasks"]
        if placements[task["id"]] == "FPGA"
    )
    return {
        "offloaded_linear_tasks": offloaded,
        "cross_device_edges": len(cross_edges),
        "uncompressed_cross_device_bytes": uncompressed_bytes,
        "logical_cross_device_bytes": logical_bytes,
        "physical_link_bytes": logical_bytes * len(cost_model.transfer_stages("GPU")),
    }


def resource_interval_us(schedule):
    totals = {"GPU": 0.0, "FPGA": 0.0}
    for row in schedule["task_rows"]:
        totals[row["device"]] = totals.get(row["device"], 0.0) + row["duration_us"]
    for row in schedule["transfer_rows"]:
        totals[row["resource"]] = totals.get(row["resource"], 0.0) + (
            row["finish_us"] - row["start_us"]
        )
    return max(totals.values()) if totals else schedule["makespan_us"]


def analytical_multi_request(schedule, request_count):
    interval = resource_interval_us(schedule)
    makespan = schedule["makespan_us"] + max(0, request_count - 1) * interval
    return {
        "makespan_us": makespan,
        "mean_completion_us": schedule["makespan_us"] + max(0, request_count - 1) * interval / 2.0,
        "throughput_requests_per_s": request_count * 1_000_000.0 / makespan,
    }


def run_analytical_final(dag, profile, optimization_config, design_config, dynamic_config, output):
    simgrid_root = Path(__file__).resolve().parent.parent
    runtime = load_module("full_dag_runtime", simgrid_root / "12_full_dag_baselines" / "run_policy.py")
    opt = load_module(
        "optimization_ablation",
        simgrid_root / "14_optimization_ablation" / "run_optimization_ablation.py",
    )
    design = load_module(
        "design_space",
        simgrid_root / "13_design_space_and_pipeline" / "run_design_space.py",
    )

    dag_data = load_json(dag)
    base_profile = load_json(profile)
    opt_config = load_json(optimization_config)
    design_cfg = load_json(design_config)
    dynamic_cfg = load_json(dynamic_config)

    ablation_output = output / "optimization_ablation"
    design_output = output / "design_space"
    dynamic_output = output / "dynamic_load"
    threshold_output = output / "adaptive_threshold"
    measurement_output = output / "measurement_plan"
    for directory in (ablation_output, design_output, dynamic_output, threshold_output):
        directory.mkdir(parents=True, exist_ok=True)
    (ablation_output / "dags").mkdir(exist_ok=True)
    (ablation_output / "profiles").mkdir(exist_ok=True)

    fused_dag = opt.build_fused_ffn_dag(dag_data, runtime)
    original_path = ablation_output / "dags" / "linear_only.json"
    fused_path = ablation_output / "dags" / "ffn_fused.json"
    original_path.write_text(json.dumps(dag_data, indent=2), encoding="utf-8")
    fused_path.write_text(json.dumps(fused_dag, indent=2), encoding="utf-8")

    ablation_rows = []
    for case in opt_config["cases"]:
        case_profile = opt.make_profile(base_profile, opt_config, case)
        profile_path = ablation_output / "profiles" / f"{case['name']}.json"
        profile_path.write_text(json.dumps(case_profile, indent=2), encoding="utf-8")
        case_dag = fused_dag if case["fusion"] else dag_data
        tasks, incoming, outgoing = runtime.graph_indexes(case_dag)
        order = runtime.topological_order(case_dag["tasks"], case_dag["edges"])
        cost_model = runtime.SensitivityCostModel(case_profile)
        schedules = {
            policy: runtime.analytical_schedule(policy, order, tasks, incoming, outgoing, cost_model)
            for policy in opt_config["policies"]
        }
        for request_count in opt_config["request_counts"]:
            gpu_multi = analytical_multi_request(schedules["gpu_only"], request_count)
            for policy, schedule in schedules.items():
                multi = analytical_multi_request(schedule, request_count)
                metrics = placement_metrics(runtime, case_dag, cost_model, schedule["placements"])
                ablation_rows.append(
                    {
                        "case": case["name"],
                        "fusion": case["fusion"],
                        "path_kind": case["path_kind"],
                        "compression": case["compression"],
                        "compression_ratio": case_profile["activation_compression"]["ratio"],
                        "fpga_input_buffers": case["fpga_input_buffers"],
                        "policy": policy,
                        "request_count": request_count,
                        "tasks_per_request": len(case_dag["tasks"]),
                        "simgrid_makespan_us": multi["makespan_us"],
                        "mean_completion_us": multi["mean_completion_us"],
                        "throughput_requests_per_s": multi["throughput_requests_per_s"],
                        **metrics,
                        "endpoint_codec_us_per_request": 0.0,
                        "task_records": len(case_dag["tasks"]) * request_count,
                        "transfer_records": metrics["cross_device_edges"]
                        * len(cost_model.transfer_stages("GPU"))
                        * request_count,
                        "valid_for_target_prediction": False,
                        "execution_mode": "analytical_fallback_no_simgrid",
                    }
                )
    gpu_baselines = {
        (row["case"], row["request_count"]): row
        for row in ablation_rows
        if row["policy"] == "gpu_only"
    }
    for row in ablation_rows:
        gpu = gpu_baselines[(row["case"], row["request_count"])]
        row["latency_speedup_vs_gpu_only"] = (
            gpu["simgrid_makespan_us"] / row["simgrid_makespan_us"]
        )
        row["throughput_speedup_vs_gpu_only"] = (
            row["throughput_requests_per_s"] / gpu["throughput_requests_per_s"]
        )
    write_csv(ablation_output / "optimization_ablation.csv", ablation_rows)
    write_json(
        ablation_output / "summary.json",
        {
            "valid_for_target_prediction": False,
            "execution_mode": "analytical_fallback_no_simgrid",
            "rows": len(ablation_rows),
            "fused_dag_tasks": len(fused_dag["tasks"]),
            "fused_dag_edges": len(fused_dag["edges"]),
        },
    )

    sweep_rows = design.run_analytical_sweep(runtime, dag_data, base_profile, design_cfg)
    threshold_rows = design.build_threshold_rows(sweep_rows, design_cfg)
    path_rows = design.summarize_paths(sweep_rows, design_cfg)
    write_csv(design_output / "design_space_sweep.csv", sweep_rows)
    write_csv(design_output / "bandwidth_thresholds.csv", threshold_rows)
    write_csv(design_output / "path_summary.csv", path_rows)
    best_sweep = max(
        (row for row in sweep_rows if row["policy"] == "communication_eft"),
        key=lambda row: row["speedup_vs_gpu_only"],
    )
    write_json(
        design_output / "summary.json",
        {
            "profile_kind": design_cfg["profile_kind"],
            "valid_for_target_prediction": False,
            "execution_mode": "analytical_fallback_no_simgrid",
            "hardware_conditions": len(sweep_rows) // len(design.POLICIES),
            "sweep_rows": len(sweep_rows),
            "threshold_rows": len(threshold_rows),
            "best_analytical_eft_condition": best_sweep,
            "best_pipeline_at_max_tested_requests": {},
        },
    )

    dynamic_profile = load_json(
        ablation_output / "profiles" / "rf880_fusion_compression_shared_double_buffer.json"
    )
    cost_model = runtime.SensitivityCostModel(dynamic_profile)
    tasks, incoming, outgoing = runtime.graph_indexes(fused_dag)
    order = runtime.topological_order(fused_dag["tasks"], fused_dag["edges"])
    mode_schedules = {
        "gpu_only": runtime.analytical_schedule(
            "gpu_only", order, tasks, incoming, outgoing, cost_model
        ),
        "latency_eft": runtime.analytical_schedule(
            "communication_eft", order, tasks, incoming, outgoing, cost_model
        ),
        "throughput_all": runtime.analytical_schedule(
            "static_all_fpga", order, tasks, incoming, outgoing, cost_model
        ),
    }
    dynamic_rows = []
    for rate in dynamic_cfg["arrival_rates_requests_per_s"]:
        for policy in dynamic_cfg["policies"]:
            mode = "latency_eft" if policy == "adaptive_dual_mode" else policy
            schedule = mode_schedules[mode]
            interval = resource_interval_us(schedule)
            service_rate = 1_000_000.0 / interval
            utilization = min(0.98, rate / service_rate)
            queue_delay = utilization * interval / max(0.02, 2.0 * (1.0 - utilization))
            p95 = schedule["makespan_us"] + 3.0 * queue_delay
            throughput = min(float(rate), service_rate)
            dynamic_rows.append(
                {
                    "offered_rate_requests_per_s": rate,
                    "policy": policy,
                    "request_count": dynamic_cfg["request_count"],
                    "mean_latency_us": schedule["makespan_us"] + queue_delay,
                    "p50_latency_us": schedule["makespan_us"] + 0.5 * queue_delay,
                    "p95_latency_us": p95,
                    "max_latency_us": p95 * 1.2,
                    "slo_violation_rate": 1.0 if p95 > dynamic_cfg["slo_us"] else 0.0,
                    "achieved_throughput_requests_per_s": throughput,
                    "latency_mode_requests": dynamic_cfg["request_count"]
                    if policy in {"latency_eft", "adaptive_dual_mode"}
                    else 0,
                    "throughput_mode_requests": dynamic_cfg["request_count"]
                    if policy == "throughput_all"
                    else 0,
                    "gpu_only_requests": dynamic_cfg["request_count"] if policy == "gpu_only" else 0,
                    "task_records": len(fused_dag["tasks"]) * dynamic_cfg["request_count"],
                    "transfer_records": 0,
                    "valid_for_target_prediction": False,
                    "execution_mode": "analytical_fallback_no_simgrid",
                }
            )
    write_csv(dynamic_output / "dynamic_load_sweep.csv", dynamic_rows)
    write_json(
        dynamic_output / "summary.json",
        {
            "valid_for_target_prediction": False,
            "execution_mode": "analytical_fallback_no_simgrid",
            "rows": len(dynamic_rows),
        },
    )
    threshold_best = {
        str(rate): {"backlog_threshold": dynamic_cfg["adaptive_backlog_threshold"]}
        for rate in dynamic_cfg["threshold_tuning_rates_requests_per_s"]
    }
    write_json(
        threshold_output / "summary.json",
        {
            "valid_for_target_prediction": False,
            "execution_mode": "analytical_fallback_no_simgrid",
            "best_by_rate": threshold_best,
        },
    )
    run_command(
        [
            sys.executable,
            str(simgrid_root / "16_hardware_calibration" / "generate_measurement_plan.py"),
            str(fused_path),
            str(measurement_output),
        ]
    )


def write_markdown(path, summary):
    lines = [
        "# RF880-AGX Final Experiment Summary",
        "",
        "This run is a hardware-informed sensitivity experiment. It fixes the RF880 + Jetson AGX PCIe x4 topology, but it is not a target-hardware measurement until the generated measurement templates are filled with real data and converted into a complete target profile.",
        "",
        "## Scope",
        "",
        f"- DAG: {summary['inputs']['dag']}",
        f"- Base profile: {summary['inputs']['profile']}",
        f"- Output directory: {summary['output_directory']}",
        f"- Target validity: {summary['valid_for_target_prediction']}",
        "",
        "## Single-Request Ablation",
        "",
        "| Case | Policy | Latency (ms) | Speedup vs same-case GPU | FPGA Linear equivalents | Cross-device edges |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["ablation"]["single_request_rows"]:
        lines.append(
            f"| {row['case']} | {row['policy']} | {row['latency_ms']:.3f} | "
            f"{row['latency_speedup_vs_gpu_only']:.3f}x | {row['offloaded_linear_tasks']} | "
            f"{row['cross_device_edges']} |"
        )

    lines.extend(
        [
            "",
            "## Best Results",
            "",
            f"- Best single-request communication-aware case: {summary['ablation']['best_single_request']['case']}, "
            f"{summary['ablation']['best_single_request']['latency_ms']:.3f} ms, "
            f"{summary['ablation']['best_single_request']['latency_speedup_vs_gpu_only']:.3f}x vs its GPU-only baseline.",
            f"- Best 16-request throughput case: {summary['ablation']['best_multi_request']['case']} / "
            f"{summary['ablation']['best_multi_request']['policy']}, "
            f"{summary['ablation']['best_multi_request']['throughput_requests_per_s']:.3f} req/s, "
            f"{summary['ablation']['best_multi_request']['throughput_speedup_vs_gpu_only']:.3f}x.",
            f"- Best design-space analytical condition: {summary['design_space']['best_eft_condition']['path_kind']}, "
            f"{summary['design_space']['best_eft_condition']['pcie_effective_bandwidth_GBps']} GB/s, "
            f"{summary['design_space']['best_eft_condition']['one_way_fixed_latency_us']} us, "
            f"FPGA speedup {summary['design_space']['best_eft_condition']['fpga_linear_speedup_over_gpu']}x.",
            "",
            "## Dynamic Load",
            "",
            "| Offered rate (req/s) | Policy | P95 latency (ms) | SLO violation | Throughput (req/s) |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["dynamic"]["selected_rows"]:
        lines.append(
            f"| {row['offered_rate_requests_per_s']} | {row['policy']} | "
            f"{row['p95_latency_ms']:.3f} | {row['slo_violation_rate']:.3f} | "
            f"{row['achieved_throughput_requests_per_s']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Hardware Bring-Up Boundary",
            "",
            f"- Generated task-measurement rows: {summary['measurement_plan']['task_measurement_rows']}",
            f"- Generated transfer-measurement rows: {summary['measurement_plan']['transfer_measurement_rows']}",
            f"- Largest planned payload: {summary['measurement_plan']['largest_payload_bytes']} bytes",
            "",
            "The next blocking step is target measurement: confirm PCIe LnkSta, run GPU kernels, run FPGA Linear/fused kernels, measure DMA paths, then rebuild a complete target profile with `simgrid_baseline/16_hardware_calibration/validate_and_build_profile.py`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def single_request_rows(ablation_rows):
    rows = []
    for row in ablation_rows:
        if as_int(row, "request_count") != 1:
            continue
        if row["policy"] not in {"gpu_only", "communication_eft", "static_all_fpga"}:
            continue
        rows.append(
            {
                "case": row["case"],
                "policy": row["policy"],
                "latency_ms": as_float(row, "simgrid_makespan_us") / 1000.0,
                "latency_speedup_vs_gpu_only": as_float(
                    row, "latency_speedup_vs_gpu_only"
                ),
                "offloaded_linear_tasks": as_int(row, "offloaded_linear_tasks"),
                "cross_device_edges": as_int(row, "cross_device_edges"),
            }
        )
    return rows


def best_ablation_rows(ablation_rows):
    single_eft = [
        row
        for row in ablation_rows
        if row["policy"] == "communication_eft" and as_int(row, "request_count") == 1
    ]
    multi = [row for row in ablation_rows if as_int(row, "request_count") == 16]
    best_single = max(single_eft, key=lambda row: as_float(row, "latency_speedup_vs_gpu_only"))
    best_multi = max(multi, key=lambda row: as_float(row, "throughput_speedup_vs_gpu_only"))
    return {
        "best_single_request": {
            "case": best_single["case"],
            "latency_ms": as_float(best_single, "simgrid_makespan_us") / 1000.0,
            "latency_speedup_vs_gpu_only": as_float(
                best_single, "latency_speedup_vs_gpu_only"
            ),
            "offloaded_linear_tasks": as_int(best_single, "offloaded_linear_tasks"),
            "cross_device_edges": as_int(best_single, "cross_device_edges"),
        },
        "best_multi_request": {
            "case": best_multi["case"],
            "policy": best_multi["policy"],
            "throughput_requests_per_s": as_float(best_multi, "throughput_requests_per_s"),
            "throughput_speedup_vs_gpu_only": as_float(
                best_multi, "throughput_speedup_vs_gpu_only"
            ),
            "request_count": as_int(best_multi, "request_count"),
        },
    }


def selected_dynamic_rows(dynamic_rows):
    max_rate = max(as_int(row, "offered_rate_requests_per_s") for row in dynamic_rows)
    selected = []
    for row in dynamic_rows:
        rate = as_int(row, "offered_rate_requests_per_s")
        if rate not in {60, 100, max_rate}:
            continue
        if row["policy"] not in {"gpu_only", "latency_eft", "adaptive_dual_mode"}:
            continue
        selected.append(
            {
                "offered_rate_requests_per_s": rate,
                "policy": row["policy"],
                "p95_latency_ms": as_float(row, "p95_latency_us") / 1000.0,
                "slo_violation_rate": as_float(row, "slo_violation_rate"),
                "achieved_throughput_requests_per_s": as_float(
                    row, "achieved_throughput_requests_per_s"
                ),
            }
        )
    return sorted(selected, key=lambda row: (row["offered_rate_requests_per_s"], row["policy"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for the final RF880-AGX experiment artifacts.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    simgrid_root = here.parent
    dag = simgrid_root / "08_coarse_vit_dag" / "results" / "vit_batch1_coarse_dag.json"
    profile = here / "rf880_agx_sensitivity_profile.json"
    optimization_config = here / "rf880_optimization_config.json"
    design_config = here / "rf880_design_space_config.json"
    dynamic_config = here / "rf880_dynamic_config.json"

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ablation_output = output / "optimization_ablation"
    design_output = output / "design_space"
    dynamic_output = output / "dynamic_load"
    threshold_output = output / "adaptive_threshold"
    measurement_output = output / "measurement_plan"

    execution_mode = "simgrid_replay"
    if simgrid_available():
        run_command(
            [
                sys.executable,
                str(simgrid_root / "14_optimization_ablation" / "run_optimization_ablation.py"),
                str(dag),
                str(profile),
                str(optimization_config),
                str(ablation_output),
            ]
        )
        run_command(
            [
                sys.executable,
                str(simgrid_root / "13_design_space_and_pipeline" / "run_design_space.py"),
                str(dag),
                str(profile),
                str(design_config),
                str(design_output),
            ]
        )

        fused_dag = ablation_output / "dags" / "ffn_fused.json"
        dynamic_profile = (
            ablation_output / "profiles" / "rf880_fusion_compression_shared_double_buffer.json"
        )
        run_command(
            [
                sys.executable,
                str(simgrid_root / "15_dynamic_runtime" / "run_load_sweep.py"),
                str(fused_dag),
                str(dynamic_profile),
                str(dynamic_config),
                str(dynamic_output),
            ]
        )
        run_command(
            [
                sys.executable,
                str(simgrid_root / "15_dynamic_runtime" / "tune_adaptive_threshold.py"),
                str(fused_dag),
                str(dynamic_profile),
                str(dynamic_config),
                str(threshold_output),
            ]
        )
        run_command(
            [
                sys.executable,
                str(simgrid_root / "16_hardware_calibration" / "generate_measurement_plan.py"),
                str(fused_dag),
                str(measurement_output),
            ]
        )
    else:
        execution_mode = "analytical_fallback_no_simgrid"
        run_analytical_final(dag, profile, optimization_config, design_config, dynamic_config, output)

    ablation_rows = load_csv(ablation_output / "optimization_ablation.csv")
    design_summary = load_json(design_output / "summary.json")
    dynamic_rows = load_csv(dynamic_output / "dynamic_load_sweep.csv")
    threshold_summary = load_json(threshold_output / "summary.json")
    measurement_summary = load_json(measurement_output / "measurement_plan_summary.json")
    best_ablation = best_ablation_rows(ablation_rows)

    summary = {
        "valid_for_target_prediction": False,
        "execution_mode": execution_mode,
        "output_directory": str(output),
        "inputs": {
            "dag": str(dag),
            "profile": str(profile),
            "optimization_config": str(optimization_config),
            "design_config": str(design_config),
            "dynamic_config": str(dynamic_config),
        },
        "ablation": {
            "rows": len(ablation_rows),
            "single_request_rows": single_request_rows(ablation_rows),
            **best_ablation,
        },
        "design_space": {
            "hardware_conditions": design_summary["hardware_conditions"],
            "best_eft_condition": design_summary["best_analytical_eft_condition"],
            "best_pipeline_at_max_tested_requests": design_summary[
                "best_pipeline_at_max_tested_requests"
            ],
        },
        "dynamic": {
            "rows": len(dynamic_rows),
            "selected_rows": selected_dynamic_rows(dynamic_rows),
            "threshold_selection": threshold_summary["best_by_rate"],
        },
        "measurement_plan": measurement_summary,
        "artifacts": {
            "optimization_ablation": str(ablation_output),
            "design_space": str(design_output),
            "dynamic_load": str(dynamic_output),
            "adaptive_threshold": str(threshold_output),
            "measurement_plan": str(measurement_output),
        },
    }
    write_json(output / "final_summary.json", summary)
    write_markdown(output / "FINAL_RESULTS.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
