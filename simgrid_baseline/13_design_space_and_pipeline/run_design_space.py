import argparse
import copy
import csv
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


POLICIES = ("gpu_only", "static_all_fpga", "communication_eft")
RESULT_PREFIX = "RESULT_JSON "


def load_runtime(path):
    spec = importlib.util.spec_from_file_location("full_dag_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_profile(base, config, path_kind, bandwidth, fixed_latency, fpga_speedup, name):
    profile = copy.deepcopy(base)
    profile["profile_name"] = name
    profile["valid_for_target_prediction"] = False
    profile["fpga"]["linear_speedup_over_gpu"] = fpga_speedup
    link = {
        "effective_bandwidth_GBps": bandwidth,
        "fixed_latency_us": fixed_latency,
    }
    if path_kind == "host_staged_two_copy":
        profile["communication_path"] = {
            "kind": path_kind,
            "gpu_host": copy.deepcopy(config["staged_gpu_host"]),
            "host_fpga_pcie": link,
            "links_share_both_directions": True,
        }
    elif path_kind == "shared_mapped_one_copy":
        profile["communication_path"] = {
            "kind": path_kind,
            "host_fpga_pcie": link,
            "links_share_both_directions": True,
        }
    elif path_kind == "direct_dma_one_copy":
        profile["communication_path"] = {
            "kind": path_kind,
            "direct_gpu_fpga": link,
            "links_share_both_directions": True,
        }
    else:
        raise RuntimeError(f"Unknown path kind {path_kind}")
    return profile


def placement_metrics(runtime, dag, cost_model, placements):
    tasks = {task["id"]: task for task in dag["tasks"]}
    cross_edges = [
        edge
        for edge in dag["edges"]
        if placements[edge["source"]] != placements[edge["target"]]
    ]
    logical_bytes = sum(cost_model.edge_bytes(edge) for edge in cross_edges)
    offloaded = sum(
        task["task_type"] in runtime.LINEAR_TYPES and placements[task["id"]] == "FPGA"
        for task in dag["tasks"]
    )
    stages = len(cost_model.transfer_stages("GPU"))
    return offloaded, len(cross_edges), logical_bytes, logical_bytes * stages


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_analytical_sweep(runtime, dag, base, config):
    tasks, incoming, outgoing = runtime.graph_indexes(dag)
    order = runtime.topological_order(dag["tasks"], dag["edges"])
    rows = []
    for path_kind in config["path_kinds"]:
        for bandwidth in config["pcie_effective_bandwidth_GBps"]:
            for fixed_latency in config["one_way_fixed_latency_us"]:
                for fpga_speedup in config["fpga_linear_speedup_over_gpu"]:
                    name = f"sweep_{path_kind}_{bandwidth}GBps_{fixed_latency}us_{fpga_speedup}x"
                    profile = make_profile(
                        base, config, path_kind, bandwidth, fixed_latency, fpga_speedup, name
                    )
                    cost_model = runtime.SensitivityCostModel(profile)
                    schedules = {}
                    for policy in POLICIES:
                        schedules[policy] = runtime.analytical_schedule(
                            policy, order, tasks, incoming, outgoing, cost_model
                        )
                    gpu_time = schedules["gpu_only"]["makespan_us"]
                    for policy in POLICIES:
                        schedule = schedules[policy]
                        offloaded, cross_edges, logical_bytes, physical_bytes = placement_metrics(
                            runtime, dag, cost_model, schedule["placements"]
                        )
                        rows.append(
                            {
                                "path_kind": path_kind,
                                "pcie_effective_bandwidth_GBps": bandwidth,
                                "one_way_fixed_latency_us": fixed_latency,
                                "fpga_linear_speedup_over_gpu": fpga_speedup,
                                "policy": policy,
                                "analytical_makespan_us": schedule["makespan_us"],
                                "speedup_vs_gpu_only": gpu_time / schedule["makespan_us"],
                                "offloaded_linear_tasks": offloaded,
                                "cross_device_edges": cross_edges,
                                "logical_cross_device_bytes": logical_bytes,
                                "physical_link_bytes": physical_bytes,
                                "valid_for_target_prediction": False,
                            }
                        )
    return rows


def build_threshold_rows(sweep_rows, config):
    eft_rows = [row for row in sweep_rows if row["policy"] == "communication_eft"]
    grouped = defaultdict(list)
    for row in eft_rows:
        key = (
            row["path_kind"],
            row["one_way_fixed_latency_us"],
            row["fpga_linear_speedup_over_gpu"],
        )
        grouped[key].append(row)
    rows = []
    for key, candidates in sorted(grouped.items()):
        profitable = sorted(
            (
                row
                for row in candidates
                if row["speedup_vs_gpu_only"] > 1.000001
                and row["offloaded_linear_tasks"] > 0
            ),
            key=lambda row: row["pcie_effective_bandwidth_GBps"],
        )
        first = profitable[0] if profitable else None
        rows.append(
            {
                "path_kind": key[0],
                "one_way_fixed_latency_us": key[1],
                "fpga_linear_speedup_over_gpu": key[2],
                "minimum_tested_profitable_bandwidth_GBps": (
                    first["pcie_effective_bandwidth_GBps"] if first else ""
                ),
                "speedup_at_threshold": first["speedup_vs_gpu_only"] if first else "",
                "offloaded_linear_tasks_at_threshold": (
                    first["offloaded_linear_tasks"] if first else 0
                ),
                "tested_bandwidths_GBps": ",".join(
                    str(value) for value in config["pcie_effective_bandwidth_GBps"]
                ),
                "valid_for_target_prediction": False,
            }
        )
    return rows


def parse_result(completed, label):
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(f"Missing result for {label}:\n{completed.stdout}\n{completed.stderr}")


def run_replays(runtime_script, dag_path, profiles, config, output_directory):
    rows = []
    for scenario_name, profile_path in profiles.items():
        for request_count in config["request_counts"]:
            for policy in POLICIES:
                case_output = output_directory / scenario_name / policy / f"requests_{request_count}"
                command = [
                    sys.executable,
                    str(runtime_script),
                    policy,
                    str(dag_path),
                    str(profile_path),
                    str(case_output),
                    "--requests",
                    str(request_count),
                ]
                completed = subprocess.run(command, text=True, capture_output=True, check=True)
                result = parse_result(completed, f"{scenario_name}/{policy}/{request_count}")
                expected_tasks = result["tasks_per_request"] * request_count
                expected_transfers = (
                    result["cross_device_edges"]
                    * result["communication_stages_per_cross_edge"]
                    * request_count
                )
                if result["task_records"] != expected_tasks:
                    raise RuntimeError(f"Task-record mismatch for {scenario_name}/{policy}/{request_count}")
                if result["transfer_records"] != expected_transfers:
                    raise RuntimeError(f"Transfer-record mismatch for {scenario_name}/{policy}/{request_count}")
                if result["request_completion_records"] != request_count:
                    raise RuntimeError(f"Completion-record mismatch for {scenario_name}/{policy}/{request_count}")
                rows.append(
                    {
                        "scenario": scenario_name,
                        "path_kind": result["communication_path_kind"],
                        "policy": policy,
                        "request_count": request_count,
                        "simgrid_makespan_us": result["simgrid_makespan_us"],
                        "mean_completion_us": result["mean_completion_us"],
                        "throughput_requests_per_s": result["throughput_requests_per_s"],
                        "offloaded_linear_tasks": result["offloaded_linear_tasks"],
                        "cross_device_edges_per_request": result["cross_device_edges"],
                        "total_physical_link_bytes": result["total_physical_link_bytes"],
                        "task_records": result["task_records"],
                        "transfer_records": result["transfer_records"],
                        "valid_for_target_prediction": False,
                    }
                )

    gpu_throughput = {
        (row["scenario"], row["request_count"]): row["throughput_requests_per_s"]
        for row in rows
        if row["policy"] == "gpu_only"
    }
    for row in rows:
        baseline = gpu_throughput[(row["scenario"], row["request_count"])]
        row["throughput_speedup_vs_gpu_only"] = row["throughput_requests_per_s"] / baseline
    return rows


def summarize_paths(sweep_rows, config):
    rows = []
    total_conditions = (
        len(config["pcie_effective_bandwidth_GBps"])
        * len(config["one_way_fixed_latency_us"])
        * len(config["fpga_linear_speedup_over_gpu"])
    )
    for path_kind in config["path_kinds"]:
        eft = [
            row
            for row in sweep_rows
            if row["path_kind"] == path_kind and row["policy"] == "communication_eft"
        ]
        static = [
            row
            for row in sweep_rows
            if row["path_kind"] == path_kind and row["policy"] == "static_all_fpga"
        ]
        rows.append(
            {
                "path_kind": path_kind,
                "tested_conditions": total_conditions,
                "eft_profitable_conditions": sum(row["speedup_vs_gpu_only"] > 1.000001 for row in eft),
                "static_profitable_conditions": sum(
                    row["speedup_vs_gpu_only"] > 1.000001 for row in static
                ),
                "eft_max_speedup": max(row["speedup_vs_gpu_only"] for row in eft),
                "eft_max_offloaded_linear_tasks": max(row["offloaded_linear_tasks"] for row in eft),
                "valid_for_target_prediction": False,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dag", type=Path)
    parser.add_argument("base_profile", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    runtime_script = here.parent / "12_full_dag_baselines" / "run_policy.py"
    runtime = load_runtime(runtime_script)
    dag_path = args.dag.resolve()
    dag = json.loads(dag_path.read_text(encoding="utf-8"))
    base = json.loads(args.base_profile.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("valid_for_target_prediction") is not False:
        raise RuntimeError("Design-space config must be invalid for target prediction")

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles_dir = output / "profiles"
    profiles_dir.mkdir(exist_ok=True)

    sweep_rows = run_analytical_sweep(runtime, dag, base, config)
    write_csv(output / "design_space_sweep.csv", sweep_rows)
    threshold_rows = build_threshold_rows(sweep_rows, config)
    write_csv(output / "bandwidth_thresholds.csv", threshold_rows)
    path_rows = summarize_paths(sweep_rows, config)
    write_csv(output / "path_summary.csv", path_rows)

    representative_profiles = {}
    for scenario in config["representative_scenarios"]:
        profile = make_profile(
            base,
            config,
            scenario["path_kind"],
            scenario["pcie_effective_bandwidth_GBps"],
            scenario["one_way_fixed_latency_us"],
            scenario["fpga_linear_speedup_over_gpu"],
            scenario["name"],
        )
        profile_path = profiles_dir / f"{scenario['name']}.json"
        profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        representative_profiles[scenario["name"]] = profile_path

    replay_rows = run_replays(
        runtime_script, dag_path, representative_profiles, config, output / "replay"
    )
    write_csv(output / "pipeline_replay.csv", replay_rows)

    eft_sweep = [row for row in sweep_rows if row["policy"] == "communication_eft"]
    best_sweep = max(eft_sweep, key=lambda row: row["speedup_vs_gpu_only"])
    request_max = max(config["request_counts"])
    max_request_rows = [row for row in replay_rows if row["request_count"] == request_max]
    best_pipeline = max(
        max_request_rows, key=lambda row: row["throughput_speedup_vs_gpu_only"]
    )
    summary = {
        "profile_kind": config["profile_kind"],
        "valid_for_target_prediction": False,
        "hardware_conditions": len(sweep_rows) // len(POLICIES),
        "sweep_rows": len(sweep_rows),
        "threshold_rows": len(threshold_rows),
        "replay_rows": len(replay_rows),
        "path_summary": path_rows,
        "best_analytical_eft_condition": best_sweep,
        "best_pipeline_at_max_tested_requests": best_pipeline,
        "artifacts": {
            "design_space_sweep": "design_space_sweep.csv",
            "bandwidth_thresholds": "bandwidth_thresholds.csv",
            "path_summary": "path_summary.csv",
            "pipeline_replay": "pipeline_replay.csv",
            "profiles": "profiles/",
            "replay": "replay/"
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
