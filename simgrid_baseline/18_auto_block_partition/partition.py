"""Dependency-checked FFN block partitioning under a fixed sensitivity profile."""

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "12_full_dag_baselines" / "run_policy.py"
DAG_PATH = ROOT / "08_coarse_vit_dag" / "results" / "vit_batch1_coarse_dag.json"
PROFILE_PATH = ROOT / "17_rf880_agx_final" / "rf880_agx_sensitivity_profile.json"
CAPABILITY_PATH = Path(__file__).with_name("capability_assumptions.json")


def load_runtime():
    spec = importlib.util.spec_from_file_location("vit_partition_runtime", POLICY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ffns_from_dependencies(dag):
    """Find three-node FFNs by edge structure, with fixed GPU interfaces."""
    tasks = {task["id"]: task for task in dag["tasks"]}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for edge in dag["edges"]:
        incoming[edge["target"]].append(edge)
        outgoing[edge["source"]].append(edge)
    regions = []
    rejected = []
    used = set()
    for task in dag["tasks"]:
        if task["task_type"] != "linear_ffn1":
            continue
        chain = [task["id"]]
        reason = None
        if len(incoming[chain[0]]) != 1 or len(outgoing[chain[0]]) != 1:
            reason = "fc1 requires exactly one input and one internal consumer"
        else:
            gelu_id = outgoing[chain[0]][0]["target"]
            chain.append(gelu_id)
            if tasks[gelu_id]["task_type"] != "gelu":
                reason = "fc1 successor is not GELU"
            elif len(incoming[gelu_id]) != 1 or len(outgoing[gelu_id]) != 1:
                reason = "GELU has an external input or consumer"
            else:
                fc2_id = outgoing[gelu_id][0]["target"]
                chain.append(fc2_id)
                if tasks[fc2_id]["task_type"] != "linear_ffn2":
                    reason = "GELU successor is not fc2"
                elif len(incoming[fc2_id]) != 1 or len(outgoing[fc2_id]) != 1:
                    reason = "fc2 requires one internal input and one GPU output"
        if reason:
            rejected.append({"root": task["id"], "reason": reason})
            continue
        if set(chain) & used:
            rejected.append({"root": task["id"], "reason": "region overlaps another region"})
            continue
        entrance = incoming[chain[0]][0]
        exit_edge = outgoing[chain[-1]][0]
        if entrance["source"] in chain or exit_edge["target"] in chain:
            rejected.append({"root": task["id"], "reason": "invalid external interface"})
            continue
        if tasks[entrance["source"]]["candidate_devices"] != ["GPU"] or tasks[exit_edge["target"]]["candidate_devices"] != ["GPU"]:
            rejected.append({"root": task["id"], "reason": "external interfaces are not fixed on GPU"})
            continue
        region = {
            "id": chain[0].rsplit("_", 1)[0],
            "task_ids": chain,
            "input_edge": entrance,
            "internal_edges": [outgoing[chain[0]][0], outgoing[chain[1]][0]],
            "output_edge": exit_edge,
        }
        regions.append(region)
        used.update(chain)
    return regions, rejected


def workspace_bytes(tasks, profile):
    element_bytes = profile["activation"]["bytes_per_element"]
    # Conservative activation footprint; not an RF880 memory measurement.
    return element_bytes * (math.prod(tasks[0]["input_shape_batch1"]) +
                            sum(math.prod(task["output_shape_batch1"]) for task in tasks))


def block_task(tasks, runtime):
    return {
        "id": tasks[0]["id"].rsplit("_", 1)[0] + "_ffn_island",
        "task_type": runtime.FUSED_FFN_TYPE,
        "candidate_devices": ["GPU", "FPGA"],
        "input_shape_batch1": tasks[0]["input_shape_batch1"],
        "output_shape_batch1": tasks[-1]["output_shape_batch1"],
        "source_onnx_node_ids": sum((t["source_onnx_node_ids"] for t in tasks), []),
        "fused_components": tasks,
    }


def candidate_cost(tasks, device, capability, model, runtime):
    types = tuple(task["task_type"] for task in tasks)
    if device == "GPU":
        return sum(model.latency_us(task, "GPU") for task in tasks), None
    available = capability["available_memory_bytes"] - capability["reserved_weight_bytes"]
    if workspace_bytes(tasks, model.profile) > min(capability["max_workspace_bytes"], available):
        return None, "block workspace exceeds assumed capacity"
    if any("FPGA" not in task["candidate_devices"] for task in tasks if task["task_type"] != "gelu"):
        return None, "task has no FPGA implementation"
    if len(tasks) == 1 and types[0] in capability["supported_fpga_task_types"]:
        return model.latency_us(tasks[0], "FPGA"), None
    if list(types) in capability["supported_fpga_blocks"] and types == ("linear_ffn1", "gelu", "linear_ffn2"):
        return model.latency_us(block_task(tasks, runtime), "FPGA"), None
    return None, "no declared FPGA block implementation"


def transfer_us(edge, source, target, model):
    if source == target:
        return 0.0
    byte_count = model.edge_bytes(edge)
    stages = sum(
        model.stage_latency_us(resource, byte_count)
        for resource, _direction in model.transfer_stages(source)
    )
    return (
        model.codec_latency_us(edge, "encode")
        + stages
        + model.codec_latency_us(edge, "decode")
    )


def solve_region(region, tasks_by_id, model, capability, runtime):
    tasks = [tasks_by_id[task_id] for task_id in region["task_ids"]]
    edges = [region["input_edge"], *region["internal_edges"], region["output_edge"]]
    n = len(tasks)
    candidates = {}
    rejected = []
    for i in range(n):
        for j in range(i + 1, n + 1):
            for device in ("GPU", "FPGA"):
                cost, reason = candidate_cost(tasks[i:j], device, capability, model, runtime)
                if reason:
                    rejected.append({"start": i, "end": j, "device": device, "reason": reason})
                else:
                    candidates[(i, j, device)] = cost
    states = {(0, "GPU"): (0.0, ())}
    for j in range(1, n + 1):
        for device in ("GPU", "FPGA"):
            options = []
            for i in range(j):
                cost = candidates.get((i, j, device))
                if cost is None:
                    continue
                for previous in ("GPU", "FPGA"):
                    if (i, previous) not in states:
                        continue
                    prior_cost, prior_blocks = states[(i, previous)]
                    block = {"task_ids": region["task_ids"][i:j], "device": device,
                             "compute_us": cost, "input_transfer_us": transfer_us(edges[i], previous, device, model)}
                    options.append((prior_cost + block["input_transfer_us"] + cost, prior_blocks + (block,)))
            if options:
                states[(j, device)] = min(options, key=lambda option: (option[0], len(option[1])))
    finals = []
    for device in ("GPU", "FPGA"):
        if (n, device) in states:
            cost, blocks = states[(n, device)]
            finals.append((cost + transfer_us(edges[n], device, "GPU", model), blocks, device))
    if not finals:
        raise RuntimeError(f"No legal partition for {region['id']}")
    cost, blocks, terminal = min(finals, key=lambda option: (option[0], len(option[1])))
    gpu_interval = sum(model.latency_us(task, "GPU") for task in tasks)
    full_fpga_compute, full_fpga_reason = candidate_cost(tasks, "FPGA", capability, model, runtime)
    full_fpga_interval = None if full_fpga_compute is None else (
        transfer_us(edges[0], "GPU", "FPGA", model) + full_fpga_compute
        + transfer_us(edges[-1], "FPGA", "GPU", model)
    )
    return {"id": region["id"], "task_ids": region["task_ids"], "interval_cost_us": cost,
            "blocks": list(blocks), "terminal_device": terminal, "rejected_candidates": rejected,
            "candidate_count": len(candidates), "gpu_interval_us": gpu_interval,
            "full_fpga_compute_us": full_fpga_compute,
            "full_fpga_boundary_transfer_us": None if full_fpga_compute is None else
            full_fpga_interval - full_fpga_compute,
            "full_fpga_interval_us": full_fpga_interval,
            "full_fpga_net_gain_us": None if full_fpga_interval is None else
            gpu_interval - full_fpga_interval,
            "full_fpga_rejected_reason": full_fpga_reason}


def quotient_dag(dag, region_solutions, runtime):
    """Contract only selected full FFNs; retain all external DAG edges."""
    original = {task["id"]: task for task in dag["tasks"]}
    replacements = {}
    grouped = []
    for solution in region_solutions:
        for block in solution["blocks"]:
            if block["device"] != "FPGA" or len(block["task_ids"]) == 1:
                continue
            components = [original[task_id] for task_id in block["task_ids"]]
            if [task["task_type"] for task in components] != ["linear_ffn1", "gelu", "linear_ffn2"]:
                raise RuntimeError("Only declared full FFN blocks can be contracted")
            island = block_task(components, runtime)
            grouped.append(island)
            for task_id in block["task_ids"]:
                if task_id in replacements:
                    raise RuntimeError("Overlapping contracted blocks")
                replacements[task_id] = island["id"]
    islands = {task["id"]: task for task in grouped}
    new_tasks = []
    inserted = set()
    for task in dag["tasks"]:
        target = replacements.get(task["id"])
        if target is None:
            new_tasks.append(copy.deepcopy(task))
        elif target not in inserted:
            new_tasks.append(islands[target])
            inserted.add(target)
    new_edges = []
    seen = set()
    for edge in dag["edges"]:
        updated = copy.deepcopy(edge)
        updated["source"] = replacements.get(edge["source"], edge["source"])
        updated["target"] = replacements.get(edge["target"], edge["target"])
        if updated["source"] == updated["target"]:
            continue
        key = (updated["source"], updated["target"], updated["tensor_role"])
        if key not in seen:
            new_edges.append(updated)
            seen.add(key)
    new_dag = copy.deepcopy(dag)
    new_dag["tasks"] = new_tasks
    new_dag["edges"] = new_edges
    runtime.topological_order(new_tasks, new_edges)
    return new_dag, replacements


def fixed_placements(dag, solutions, replacements):
    placements = {task["id"]: "GPU" for task in dag["tasks"]}
    for solution in solutions:
        for block in solution["blocks"]:
            target = replacements.get(block["task_ids"][0], block["task_ids"][0])
            placements[target] = block["device"]
    return placements


def split_ffn_placements(dag, regions):
    placements = {task["id"]: "GPU" for task in dag["tasks"]}
    for region in regions:
        placements[region["task_ids"][0]] = "FPGA"
        placements[region["task_ids"][2]] = "FPGA"
    return placements


def forced_full_ffn(dag, solutions, runtime):
    forced = copy.deepcopy(solutions)
    for solution in forced:
        if solution["full_fpga_compute_us"] is None:
            continue
        solution["blocks"] = [{"task_ids": solution["task_ids"], "device": "FPGA",
                                "compute_us": solution["full_fpga_compute_us"]}]
    quotient, replacements = quotient_dag(dag, forced, runtime)
    return quotient, fixed_placements(quotient, forced, replacements)


def declared_singleton_dag(dag, capability, model, runtime):
    restricted = copy.deepcopy(dag)
    for task in restricted["tasks"]:
        if "FPGA" not in task["candidate_devices"]:
            continue
        cost, _reason = candidate_cost([task], "FPGA", capability, model, runtime)
        if cost is None:
            task["candidate_devices"] = ["GPU"]
    return restricted


def evaluate_fixed(dag, placements, model, runtime):
    tasks, incoming, outgoing = runtime.graph_indexes(dag)
    order = runtime.topological_order(dag["tasks"], dag["edges"])
    device_available = {"GPU": 0.0, "FPGA": 0.0}
    link_available = {resource: 0.0 for resource in model.link_resources()}
    finish_times = {}
    transfers = []
    task_rows = []
    for task_id in order:
        device = placements[task_id]
        option = runtime.evaluate_candidate(task_id, device, tasks, incoming, outgoing,
                                            placements, finish_times, device_available,
                                            link_available, model)
        finish_times[task_id] = option["finish_us"]
        device_available[device] = option["finish_us"]
        link_available = option["link_available"]
        transfers.extend(option["transfers"])
        task_rows.append({"task_id": task_id, "device": device,
                          "start_us": option["start_us"], "finish_us": option["finish_us"]})
    cross_edges = [edge for edge in dag["edges"]
                   if placements[edge["source"]] != placements[edge["target"]]]
    return {"makespan_us": max(finish_times.values()), "cross_device_edges": len(cross_edges),
            "logical_cross_device_bytes": sum(model.edge_bytes(edge) for edge in cross_edges),
            "physical_transfer_stages": len(transfers), "task_timeline": task_rows,
            "transfer_timeline": transfers, "placements": placements}


def baseline(dag, model, runtime, policy):
    tasks, incoming, outgoing = runtime.graph_indexes(dag)
    order = runtime.topological_order(dag["tasks"], dag["edges"])
    schedule = runtime.analytical_schedule(policy, order, tasks, incoming, outgoing, model)
    return evaluate_fixed(dag, schedule["placements"], model, runtime)


def run(dag, profile, capability, runtime):
    if (profile["profile_kind"] == "target_measurement" and
            capability["status"] != "verified_target_capability"):
        raise RuntimeError("Target profile requires verified FPGA capability")
    if capability["activation_precision"] != profile["activation"]["precision"]:
        raise RuntimeError("Capability precision and profile precision differ")
    if capability["available_memory_bytes"] <= 0 or capability["max_workspace_bytes"] <= 0:
        raise RuntimeError("Capability memory limits must be positive")
    if capability["reserved_weight_bytes"] < 0:
        raise RuntimeError("Reserved weight bytes must be nonnegative")
    if capability["reserved_weight_bytes"] > capability["available_memory_bytes"]:
        raise RuntimeError("Reserved weights exceed assumed available memory")
    model = runtime.SensitivityCostModel(profile)
    regions, rejected_regions = ffns_from_dependencies(dag)
    tasks = {task["id"]: task for task in dag["tasks"]}
    started = time.perf_counter()
    solutions = [solve_region(region, tasks, model, capability, runtime) for region in regions]
    solver_ms = (time.perf_counter() - started) * 1000.0
    quotient, replacements = quotient_dag(dag, solutions, runtime)
    chosen = evaluate_fixed(quotient, fixed_placements(quotient, solutions, replacements), model, runtime)
    gpu = baseline(dag, model, runtime, "gpu_only")
    declared_dag = declared_singleton_dag(dag, capability, model, runtime)
    declared_static = baseline(declared_dag, model, runtime, "static_all_fpga")
    declared_eft = baseline(declared_dag, model, runtime, "communication_eft")
    counterfactual = {}
    if profile["profile_kind"] == "sensitivity":
        counterfactual = {
            "static_all_linear_fpga": baseline(dag, model, runtime, "static_all_fpga"),
            "operator_eft": baseline(dag, model, runtime, "communication_eft"),
        }
    split = evaluate_fixed(dag, split_ffn_placements(dag, regions), model, runtime)
    fixed_dag, fixed_placement = forced_full_ffn(dag, solutions, runtime)
    fixed = evaluate_fixed(fixed_dag, fixed_placement, model, runtime)
    compared = (("gpu_only", gpu), ("static_declared_singletons_fpga", declared_static),
                ("operator_eft_declared", declared_eft),
                ("split_ffn_linears", split), ("fixed_ffn", fixed),
                ("automatic_region_dp", chosen), *counterfactual.items())
    summaries = {name: {key: value for key, value in result.items()
                        if key not in {"placements", "task_timeline", "transfer_timeline"}}
                 for name, result in compared}
    return {"valid_for_target_prediction": profile["valid_for_target_prediction"],
            "execution_mode": "analytical_fixed_placement_replay_no_simgrid",
            "capability_status": capability["status"],
            "cost_accounting": "each task once; each device-crossing DAG edge once in full replay; DP charges only region boundary crossings",
            "region_objective": "sum of serial interval costs, not full-DAG makespan",
            "regions": solutions,
            "rejected_regions": rejected_regions, "solver_time_ms": solver_ms,
            "comparison": summaries, "mixed_dag": quotient,
            "placement_records": chosen["placements"],
            "task_timeline": chosen["task_timeline"],
            "transfer_timeline": chosen["transfer_timeline"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dag", type=Path, default=DAG_PATH)
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH)
    parser.add_argument("--capability", type=Path, default=CAPABILITY_PATH)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results") / "default")
    parser.add_argument("--simgrid-requests", type=int, default=0,
                        help="Optional actual SimGrid replay; requires Python bindings")
    parser.add_argument("--simgrid-buffers", type=int, default=1,
                        help="FPGA input buffer capacity for optional replay")
    parser.add_argument("--simgrid-policy", choices=("automatic_region_dp", "gpu_only",
                                                     "static_all_linear_fpga",
                                                     "split_ffn_linears", "fixed_ffn"),
                        default="automatic_region_dp")
    args = parser.parse_args()
    runtime = load_runtime()
    if args.simgrid_requests and runtime.Engine is None:
        raise RuntimeError("SimGrid replay requested, but Python simgrid bindings are absent")
    if args.simgrid_requests < 0 or args.simgrid_buffers <= 0:
        raise RuntimeError("Request count must be nonnegative and buffer count positive")
    dag = json.loads(args.dag.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    capability = json.loads(args.capability.read_text(encoding="utf-8"))
    result = run(dag, profile, capability, runtime)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, key in (("summary.json", None), ("mixed_dag.json", "mixed_dag"),
                      ("placements.json", "placement_records"),
                      ("task_timeline.json", "task_timeline"),
                      ("transfer_timeline.json", "transfer_timeline")):
        content = result if key is None else result[key]
        if key is None:
            content = {item: value for item, value in result.items()
                       if item not in {"mixed_dag", "placement_records", "task_timeline", "transfer_timeline"}}
        (args.output / name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    if args.simgrid_requests:
        replay_profile = copy.deepcopy(profile)
        replay_profile["pipeline"]["fpga_input_buffers"] = args.simgrid_buffers
        replay_model = runtime.SensitivityCostModel(replay_profile)
        platform = args.output / "simgrid_platform.xml"
        runtime.write_platform(replay_profile, platform)
        if args.simgrid_policy == "gpu_only":
            replay_dag = dag
            replay_placements = {task["id"]: "GPU" for task in dag["tasks"]}
        elif args.simgrid_policy == "static_all_linear_fpga":
            replay_dag = dag
            task_index, incoming, outgoing = runtime.graph_indexes(dag)
            order = runtime.topological_order(dag["tasks"], dag["edges"])
            replay_placements = runtime.analytical_schedule(
                "static_all_fpga", order, task_index, incoming, outgoing,
                replay_model)["placements"]
        elif args.simgrid_policy == "split_ffn_linears":
            replay_dag = dag
            replay_regions, _ = ffns_from_dependencies(dag)
            replay_placements = split_ffn_placements(dag, replay_regions)
        elif args.simgrid_policy == "fixed_ffn":
            replay_dag, replay_placements = forced_full_ffn(
                dag, result["regions"], runtime)
        else:
            replay_dag = result["mixed_dag"]
            replay_placements = result["placement_records"]
        makespan, task_rows, transfer_rows, completions = runtime.run_simgrid(
            replay_dag, replay_model, replay_placements, platform,
            request_count=args.simgrid_requests)
        replay = {"policy": args.simgrid_policy, "buffers": args.simgrid_buffers,
                  "requests": args.simgrid_requests, "makespan_us": makespan,
                  "request_completion_us": completions, "task_count": len(task_rows),
                  "transfer_stage_count": len(transfer_rows),
                  "execution_mode": ("simgrid_event_replay_target_measurement_profile"
                                     if profile["profile_kind"] == "target_measurement" else
                                     "simgrid_event_replay_sensitivity_parameters"),
                  "valid_for_target_prediction": profile["valid_for_target_prediction"],
                  "model_variant": dag["metadata"].get("model_variant"),
                  "activation_precision": profile["activation"]["precision"],
                  "profile_name": profile["profile_name"]}
        (args.output / "simgrid_replay_summary.json").write_text(
            json.dumps(replay, indent=2) + "\n", encoding="utf-8")
        (args.output / "simgrid_trace.json").write_text(
            json.dumps({"tasks": task_rows, "transfers": transfer_rows}, indent=2)
            + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "comparison": result["comparison"],
                      "regions": len(result["regions"]), "mode": result["execution_mode"]}, indent=2))


if __name__ == "__main__":
    main()
