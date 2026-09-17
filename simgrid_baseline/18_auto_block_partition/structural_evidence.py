"""Reproducible FFN boundary analysis and restricted alternative comparisons."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import partition


LINEAR_CLASSES = (
    "linear_qkv", "linear_projection", "linear_ffn1",
    "linear_ffn2", "linear_classifier",
)


def placement_result(dag, model, runtime, fpga_task_ids):
    placements = {task["id"]: ("FPGA" if task["id"] in fpga_task_ids else "GPU")
                  for task in dag["tasks"]}
    return partition.evaluate_fixed(dag, placements, model, runtime)


def summarize(replay):
    return {key: replay[key] for key in (
        "makespan_us", "cross_device_edges", "logical_cross_device_bytes",
        "physical_transfer_stages",
    )}


def ffn_boundary(region, model):
    input_edge = region["input_edge"]
    first, second = region["internal_edges"]
    output_edge = region["output_edge"]
    sizes = [model.uncompressed_edge_bytes(edge)
             for edge in (input_edge, first, second, output_edge)]
    split_bytes = sum(sizes)
    fused_bytes = sizes[0] + sizes[-1]
    split_transfer_us = sum(
        partition.transfer_us(edge, source, target, model)
        for edge, source, target in zip(
            (input_edge, first, second, output_edge),
            ("GPU", "FPGA", "GPU", "FPGA"),
            ("FPGA", "GPU", "FPGA", "GPU"),
        )
    )
    fused_transfer_us = (partition.transfer_us(input_edge, "GPU", "FPGA", model)
                         + partition.transfer_us(output_edge, "FPGA", "GPU", model))
    return {
        "region_id": region["id"],
        "edge_bytes_uncompressed": sizes,
        "split_logical_bytes": split_bytes,
        "fused_logical_bytes": fused_bytes,
        "logical_byte_reduction_ratio": 1.0 - fused_bytes / split_bytes,
        "split_transfer_us": split_transfer_us,
        "fused_transfer_us": fused_transfer_us,
        "saved_internal_transfer_us": split_transfer_us - fused_transfer_us,
    }


def run(dag, profile, capability, runtime):
    model = runtime.SensitivityCostModel(profile)
    regions, rejected = partition.ffns_from_dependencies(dag)
    tasks = {task["id"]: task for task in dag["tasks"]}
    boundaries = [ffn_boundary(region, model) for region in regions]
    split_ids = {task_id for region in regions
                 for task_id in (region["task_ids"][0], region["task_ids"][2])}
    base = placement_result(dag, model, runtime, set())
    split = placement_result(dag, model, runtime, split_ids)

    declared = partition.run(dag, profile, capability, runtime)
    no_full_capability = copy.deepcopy(capability)
    no_full_capability["supported_fpga_blocks"] = []
    no_full = partition.run(dag, profile, no_full_capability, runtime)
    selected_full = sum(len(solution["blocks"]) == 1
                        and solution["blocks"][0]["device"] == "FPGA"
                        and len(solution["blocks"][0]["task_ids"]) == 3
                        for solution in declared["regions"])

    classes = Counter(task["task_type"] for task in dag["tasks"])
    alternatives = {}
    for task_type in LINEAR_CLASSES:
        ids = {task["id"] for task in dag["tasks"]
               if task["task_type"] == task_type}
        if not ids:
            continue
        alternative = placement_result(dag, model, runtime, ids)
        alternatives[task_type] = {
            "candidate_count": classes[task_type],
            "scope": "counterfactual_singleton_kernel_support_not_rf880_capability",
            "assumed_fpga_tasks": len(ids),
            **summarize(alternative),
            "speedup_vs_gpu_only": base["makespan_us"] / alternative["makespan_us"],
        }

    fusion_conditions = []
    for region, boundary in zip(regions, boundaries):
        components = [tasks[task_id] for task_id in region["task_ids"]]
        fused_cost, rejected_reason = partition.candidate_cost(
            components, "FPGA", capability, model, runtime)
        split_compute = (model.latency_us(components[0], "FPGA")
                         + model.latency_us(components[1], "GPU")
                         + model.latency_us(components[2], "FPGA"))
        fusion_conditions.append({
            "region_id": region["id"],
            "fused_kernel_supported_by_declared_capability": fused_cost is not None,
            "capability_rejection": rejected_reason,
            "fused_minus_split_compute_us": None if fused_cost is None
            else fused_cost - split_compute,
            "saved_internal_transfer_us": boundary["saved_internal_transfer_us"],
            "fusion_wins_isolated_additive_cost": None if fused_cost is None
            else fused_cost - split_compute < boundary["saved_internal_transfer_us"],
            "workspace_bytes_conservative": partition.workspace_bytes(components, profile),
        })

    return {
        "status": "unmeasured_sensitivity_analysis",
        "valid_for_target_prediction": profile["valid_for_target_prediction"],
        "source_model_variant": dag["metadata"].get("model_variant"),
        "profile_name": profile["profile_name"],
        "capability_status": capability["status"],
        "ffn_regions": len(regions),
        "rejected_ffn_regions": rejected,
        "selected_full_ffns": selected_full,
        "boundary_by_ffn": boundaries,
        "fusion_condition_by_ffn": fusion_conditions,
        "full_dag_comparison": {
            "gpu_only": summarize(base),
            "split_two_linears_fpga_gelu_gpu": summarize(split),
            "declared_capability_auto": declared["comparison"]["automatic_region_dp"],
            "declared_capability_fixed_ffn": declared["comparison"]["fixed_ffn"],
            "no_full_ffn_kernel_auto": no_full["comparison"]["automatic_region_dp"],
        },
        "alternative_singleton_linear_classes": alternatives,
        "limitations": [
            "FPGA full-FFN/GELU and memory capacity are unverified capability assumptions.",
            "Alternative Linear classes are counterfactual singleton placements, not RF880 kernels.",
            "This does not search Attention or arbitrary branched subgraphs.",
            "Analytical fixed-placement replay is not a board measurement or a global DAG optimum.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dag", type=Path, default=partition.DAG_PATH)
    parser.add_argument("--profile", type=Path, default=partition.PROFILE_PATH)
    parser.add_argument("--capability", type=Path, default=partition.CAPABILITY_PATH)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("results") /
                        "structural_evidence.json")
    args = parser.parse_args()
    runtime = partition.load_runtime()
    dag = json.loads(args.dag.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    capability = json.loads(args.capability.read_text(encoding="utf-8"))
    result = run(dag, profile, capability, runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "full_dag_comparison": result["full_dag_comparison"],
                      "ffn_regions": result["ffn_regions"]}, indent=2))


if __name__ == "__main__":
    main()
