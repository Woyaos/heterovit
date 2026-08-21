import argparse
import copy
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


RESULT_PREFIX = "RESULT_JSON "


def load_runtime(path):
    spec = importlib.util.spec_from_file_location("full_dag_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_fused_ffn_dag(dag, runtime):
    fused = copy.deepcopy(dag)
    tasks_by_id = {task["id"]: task for task in fused["tasks"]}
    replacements = {}
    fused_tasks = []

    for block in range(12):
        prefix = f"block_{block:02d}"
        component_ids = [f"{prefix}_fc1", f"{prefix}_gelu", f"{prefix}_fc2"]
        components = [tasks_by_id[task_id] for task_id in component_ids]
        island_id = f"{prefix}_ffn_island"
        island = {
            "id": island_id,
            "task_type": runtime.FUSED_FFN_TYPE,
            "candidate_devices": ["GPU", "FPGA"],
            "input_shape_batch1": components[0]["input_shape_batch1"],
            "output_shape_batch1": components[-1]["output_shape_batch1"],
            "source_onnx_node_ids": sum(
                (component["source_onnx_node_ids"] for component in components), []
            ),
            "fused_components": components,
        }
        fused_tasks.append(island)
        for component_id in component_ids:
            replacements[component_id] = island_id

    new_tasks = []
    island_by_id = {task["id"]: task for task in fused_tasks}
    inserted = set()
    for task in fused["tasks"]:
        replacement = replacements.get(task["id"])
        if replacement is None:
            new_tasks.append(task)
        elif replacement not in inserted:
            new_tasks.append(island_by_id[replacement])
            inserted.add(replacement)

    new_edges = []
    seen = set()
    for edge in fused["edges"]:
        updated = copy.deepcopy(edge)
        updated["source"] = replacements.get(edge["source"], edge["source"])
        updated["target"] = replacements.get(edge["target"], edge["target"])
        if updated["source"] == updated["target"]:
            continue
        key = (updated["source"], updated["target"], updated["tensor_role"])
        if key not in seen:
            new_edges.append(updated)
            seen.add(key)

    fused["tasks"] = new_tasks
    fused["edges"] = new_edges
    fused["metadata"]["node_granularity"] = "coarse ViT tasks with fused FPGA FFN islands"
    fused["metadata"]["ffn_fusion"] = "FFN1 + GELU + FFN2"
    runtime.topological_order(new_tasks, new_edges)
    if len(fused_tasks) != 12 or len(new_tasks) != 101 or len(new_edges) != 124:
        raise RuntimeError(
            f"Unexpected fused DAG size: islands={len(fused_tasks)}, "
            f"tasks={len(new_tasks)}, edges={len(new_edges)}"
        )
    return fused


def make_profile(base, config, case):
    profile = copy.deepcopy(base)
    profile["profile_name"] = case["name"]
    profile["valid_for_target_prediction"] = False
    profile["fpga"]["elementwise_speedup_over_gpu"] = config[
        "fpga_elementwise_speedup_over_gpu"
    ]
    profile["activation_compression"] = copy.deepcopy(
        config["compression_profiles"][case["compression"]]
    )
    profile["pipeline"] = {"fpga_input_buffers": case["fpga_input_buffers"]}
    profile["optimization_case"] = copy.deepcopy(case)

    old_path = base["communication_path"]
    pcie = copy.deepcopy(old_path["host_fpga_pcie"])
    if case["path_kind"] == "host_staged_two_copy":
        profile["communication_path"] = copy.deepcopy(old_path)
    elif case["path_kind"] == "shared_mapped_one_copy":
        profile["communication_path"] = {
            "kind": "shared_mapped_one_copy",
            "host_fpga_pcie": pcie,
            "links_share_both_directions": True,
        }
    elif case["path_kind"] == "direct_dma_one_copy":
        profile["communication_path"] = {
            "kind": "direct_dma_one_copy",
            "direct_gpu_fpga": pcie,
            "links_share_both_directions": True,
        }
    else:
        raise RuntimeError(f"Unknown path kind {case['path_kind']}")
    return profile


def parse_result(completed, label):
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(f"Missing result for {label}:\n{completed.stdout}\n{completed.stderr}")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_results_markdown(path, rows):
    eft = [row for row in rows if row["policy"] == "communication_eft"]
    single = [row for row in eft if row["request_count"] == 1]
    pipeline = [row for row in eft if row["request_count"] == 8]
    lines = [
        "# Experiment 14 Results",
        "",
        "All values are sensitivity results and are invalid for target-hardware prediction.",
        "",
        "## Communication-Aware Single-Request Ablation",
        "",
        "| Case | Makespan (ms) | Speedup vs GPU | FPGA Linear equivalents | Cross edges | Encoded bytes (MB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in single:
        lines.append(
            f"| {row['case']} | {row['simgrid_makespan_us'] / 1000.0:.3f} | "
            f"{row['latency_speedup_vs_gpu_only']:.3f}x | {row['offloaded_linear_tasks']} | "
            f"{row['cross_device_edges']} | {row['logical_cross_device_bytes'] / 1e6:.3f} |"
        )
    static_single = [
        row for row in rows if row["policy"] == "static_all_fpga" and row["request_count"] == 1
    ]
    lines.extend(
        [
            "",
            "## Static-All Diagnostic",
            "",
            "| Case | Makespan (ms) | Speedup vs GPU | Cross edges | Encoded bytes (MB) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in static_single:
        lines.append(
            f"| {row['case']} | {row['simgrid_makespan_us'] / 1000.0:.3f} | "
            f"{row['latency_speedup_vs_gpu_only']:.3f}x | {row['cross_device_edges']} | "
            f"{row['logical_cross_device_bytes'] / 1e6:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Communication-Aware Eight-Request Pipeline",
            "",
            "| Case | Throughput (req/s) | Throughput speedup | Input buffers |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in pipeline:
        lines.append(
            f"| {row['case']} | {row['throughput_requests_per_s']:.3f} | "
            f"{row['throughput_speedup_vs_gpu_only']:.3f}x | {row['fpga_input_buffers']} |"
        )
    full_pipeline = [
        row
        for row in rows
        if row["case"]
        in {
            "fusion_compression_one_copy_single_buffer",
            "fusion_compression_one_copy_double_buffer",
        }
        and row["request_count"] == 8
        and row["policy"] != "gpu_only"
    ]
    lines.extend(
        [
            "",
            "## Buffer and Objective Comparison",
            "",
            "| Case | Policy | Throughput (req/s) | Speedup vs GPU |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in full_pipeline:
        lines.append(
            f"| {row['case']} | {row['policy']} | {row['throughput_requests_per_s']:.3f} | "
            f"{row['throughput_speedup_vs_gpu_only']:.3f}x |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    dag = json.loads(args.dag.read_text(encoding="utf-8"))
    base = json.loads(args.base_profile.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("valid_for_target_prediction") is not False:
        raise RuntimeError("Optimization config must declare valid_for_target_prediction=false")

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dags_dir = output / "dags"
    profiles_dir = output / "profiles"
    replay_dir = output / "replay"
    dags_dir.mkdir(exist_ok=True)
    profiles_dir.mkdir(exist_ok=True)
    replay_dir.mkdir(exist_ok=True)

    original_path = dags_dir / "linear_only.json"
    fused_path = dags_dir / "ffn_fused.json"
    original_path.write_text(json.dumps(dag, indent=2), encoding="utf-8")
    fused_dag = build_fused_ffn_dag(dag, runtime)
    fused_path.write_text(json.dumps(fused_dag, indent=2), encoding="utf-8")

    rows = []
    for case in config["cases"]:
        profile = make_profile(base, config, case)
        profile_path = profiles_dir / f"{case['name']}.json"
        profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        dag_path = fused_path if case["fusion"] else original_path
        for request_count in config["request_counts"]:
            for policy in config["policies"]:
                case_output = replay_dir / case["name"] / policy / f"requests_{request_count}"
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
                result = parse_result(completed, f"{case['name']}/{policy}/{request_count}")
                expected_tasks = result["tasks_per_request"] * request_count
                expected_transfers = (
                    result["cross_device_edges"]
                    * result["communication_stages_per_cross_edge"]
                    * request_count
                )
                if result["task_records"] != expected_tasks:
                    raise RuntimeError(f"Task record mismatch for {case['name']}/{policy}")
                if result["transfer_records"] != expected_transfers:
                    raise RuntimeError(f"Transfer record mismatch for {case['name']}/{policy}")
                rows.append(
                    {
                        "case": case["name"],
                        "fusion": case["fusion"],
                        "path_kind": case["path_kind"],
                        "compression": case["compression"],
                        "compression_ratio": result["activation_compression_ratio"],
                        "fpga_input_buffers": result["fpga_input_buffers"],
                        "policy": policy,
                        "request_count": request_count,
                        "tasks_per_request": result["tasks_per_request"],
                        "simgrid_makespan_us": result["simgrid_makespan_us"],
                        "mean_completion_us": result["mean_completion_us"],
                        "throughput_requests_per_s": result["throughput_requests_per_s"],
                        "offloaded_linear_tasks": result["offloaded_linear_tasks"],
                        "cross_device_edges": result["cross_device_edges"],
                        "uncompressed_cross_device_bytes": result[
                            "uncompressed_cross_device_bytes"
                        ],
                        "logical_cross_device_bytes": result["logical_cross_device_bytes"],
                        "physical_link_bytes_per_request": result["physical_link_bytes"],
                        "endpoint_codec_us_per_request": result["endpoint_codec_us_per_request"],
                        "task_records": result["task_records"],
                        "transfer_records": result["transfer_records"],
                        "valid_for_target_prediction": False,
                    }
                )

    gpu_baselines = {
        (row["case"], row["request_count"]): row
        for row in rows
        if row["policy"] == "gpu_only"
    }
    for row in rows:
        gpu = gpu_baselines[(row["case"], row["request_count"])]
        row["latency_speedup_vs_gpu_only"] = (
            gpu["simgrid_makespan_us"] / row["simgrid_makespan_us"]
        )
        row["throughput_speedup_vs_gpu_only"] = (
            row["throughput_requests_per_s"] / gpu["throughput_requests_per_s"]
        )

    write_csv(output / "optimization_ablation.csv", rows)
    write_results_markdown(output / "RESULTS.md", rows)
    eft_rows = [row for row in rows if row["policy"] == "communication_eft"]
    summary = {
        "profile_kind": config["profile_kind"],
        "valid_for_target_prediction": False,
        "cases": len(config["cases"]),
        "rows": len(rows),
        "fused_dag_tasks": len(fused_dag["tasks"]),
        "fused_dag_edges": len(fused_dag["edges"]),
        "best_single_request": max(
            (row for row in eft_rows if row["request_count"] == 1),
            key=lambda row: row["latency_speedup_vs_gpu_only"],
        ),
        "best_eight_request_throughput": max(
            (row for row in eft_rows if row["request_count"] == 8),
            key=lambda row: row["throughput_speedup_vs_gpu_only"],
        ),
        "best_eight_request_throughput_overall": max(
            (row for row in rows if row["request_count"] == 8),
            key=lambda row: row["throughput_speedup_vs_gpu_only"],
        ),
        "artifacts": {
            "ablation": "optimization_ablation.csv",
            "results": "RESULTS.md",
            "profiles": "profiles/",
            "dags": "dags/",
            "replay": "replay/",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
