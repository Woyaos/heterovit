import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

from simgrid import Actor, Engine, Mailbox, Mutex, Semaphore, this_actor


POLICIES = ("gpu_only", "latency_eft", "throughput_all", "adaptive_dual_mode")


def load_runtime(path):
    spec = importlib.util.spec_from_file_location("full_dag_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values, percentile_value):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[rank]


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_buffer_inputs(placements, incoming):
    for task_id, task_inputs in incoming.items():
        if placements[task_id] != "FPGA":
            continue
        cross_inputs = sum(
            placements[edge["source"]] == "GPU" for _edge_index, edge in task_inputs
        )
        if cross_inputs > 1:
            raise RuntimeError(
                f"Buffered replay supports one GPU input edge per FPGA task, got "
                f"{cross_inputs} for {task_id}"
            )


def run_case(runtime, dag, profile, policy, request_count, arrival_rate, threshold, slo_us, output):
    cost_model = runtime.SensitivityCostModel(profile)
    tasks, incoming, outgoing = runtime.graph_indexes(dag)
    order = runtime.topological_order(dag["tasks"], dag["edges"])
    schedules = {
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
    templates = {name: schedule["placements"] for name, schedule in schedules.items()}
    for placements in templates.values():
        validate_buffer_inputs(placements, incoming)

    output.mkdir(parents=True, exist_ok=True)
    platform_path = output / "generated_platform.xml"
    runtime.write_platform(profile, platform_path)

    engine = Engine(["run_dynamic_case.py"])
    Engine.set_config("network/model:CM02")
    engine.load_platform(str(platform_path))
    hosts = {name: engine.host_by_name(name) for name in ("GPU", "HOST", "FPGA")}
    device_locks = {"GPU": Mutex(), "FPGA": Mutex()}
    policy_lock = Mutex()
    fpga_buffers = Semaphore(cost_model.fpga_input_buffers or 1)
    state = {"inflight": 0}
    interval_us = 1_000_000.0 / arrival_rate
    completions = []
    decisions = []
    task_timeline = []
    transfer_timeline = []

    def target_mailbox(request_id, edge_index):
        return Mailbox.by_name(f"request-{request_id}-edge-{edge_index}-target")

    def relay_mailbox(request_id, edge_index):
        return Mailbox.by_name(f"request-{request_id}-edge-{edge_index}-relay")

    def execute_device_work(device, duration_us):
        if duration_us <= 0.0:
            return
        device_locks[device].lock()
        this_actor.execute(duration_us * 1000.0)
        device_locks[device].unlock()

    def task_actor(request_id, task_id, placements, selected_mode, arrival_us):
        task = tasks[task_id]
        device = placements[task_id]
        cross_inputs = []
        for edge_index, edge in incoming[task_id]:
            target_mailbox(request_id, edge_index).get()
            if placements[edge["source"]] != device:
                cross_inputs.append(edge)
                execute_device_work(device, cost_model.codec_latency_us(edge, "decode"))

        device_locks[device].lock()
        start = engine.clock
        this_actor.execute(cost_model.latency_us(task, device) * 1000.0)
        finish = engine.clock
        device_locks[device].unlock()
        if device == "FPGA" and cross_inputs:
            fpga_buffers.release()
        task_timeline.append(
            {
                "request_id": request_id,
                "selected_mode": selected_mode,
                "task_id": task_id,
                "task_type": task["task_type"],
                "device": device,
                "start_us": start * 1_000_000.0,
                "finish_us": finish * 1_000_000.0,
            }
        )

        if not outgoing[task_id]:
            completion_us = finish * 1_000_000.0
            policy_lock.lock()
            state["inflight"] -= 1
            policy_lock.unlock()
            completions.append(
                {
                    "request_id": request_id,
                    "selected_mode": selected_mode,
                    "arrival_us": arrival_us,
                    "completion_us": completion_us,
                    "latency_us": completion_us - arrival_us,
                    "slo_met": completion_us - arrival_us <= slo_us,
                }
            )

        for edge_index, edge in outgoing[task_id]:
            target_device = placements[edge["target"]]
            if target_device == device:
                target_mailbox(request_id, edge_index).put(edge_index, 0)
                continue
            if device == "GPU" and target_device == "FPGA":
                fpga_buffers.acquire()
            execute_device_work(device, cost_model.codec_latency_us(edge, "encode"))
            byte_count = cost_model.edge_bytes(edge)
            transfer_start = engine.clock
            if cost_model.path["kind"] == "host_staged_two_copy":
                relay_mailbox(request_id, edge_index).put(edge_index, byte_count)
            else:
                target_mailbox(request_id, edge_index).put(edge_index, byte_count)
            transfer_finish = engine.clock
            resource, direction = cost_model.transfer_stages(device)[0]
            transfer_timeline.append(
                {
                    "request_id": request_id,
                    "selected_mode": selected_mode,
                    "edge_index": edge_index,
                    "stage": 1,
                    "resource": resource,
                    "direction": direction,
                    "bytes": byte_count,
                    "start_us": transfer_start * 1_000_000.0,
                    "finish_us": transfer_finish * 1_000_000.0,
                }
            )

    def relay_actor(request_id, edge_index, edge, placements, selected_mode):
        relay_mailbox(request_id, edge_index).get()
        source_device = placements[edge["source"]]
        byte_count = cost_model.edge_bytes(edge)
        transfer_start = engine.clock
        target_mailbox(request_id, edge_index).put(edge_index, byte_count)
        transfer_finish = engine.clock
        resource, direction = cost_model.transfer_stages(source_device)[1]
        transfer_timeline.append(
            {
                "request_id": request_id,
                "selected_mode": selected_mode,
                "edge_index": edge_index,
                "stage": 2,
                "resource": resource,
                "direction": direction,
                "bytes": byte_count,
                "start_us": transfer_start * 1_000_000.0,
                "finish_us": transfer_finish * 1_000_000.0,
            }
        )

    def launch_request(request_id):
        arrival_us = request_id * interval_us
        this_actor.sleep_until(arrival_us / 1_000_000.0)
        policy_lock.lock()
        backlog = state["inflight"]
        if policy == "adaptive_dual_mode":
            selected_mode = "throughput_all" if backlog >= threshold else "latency_eft"
        else:
            selected_mode = policy
        state["inflight"] += 1
        policy_lock.unlock()
        placements = templates[selected_mode]
        decisions.append(
            {
                "request_id": request_id,
                "arrival_us": arrival_us,
                "backlog_at_arrival": backlog,
                "selected_mode": selected_mode,
            }
        )

        for task_id in tasks:
            Actor.create(
                f"request-{request_id}-task-{task_id}",
                hosts[placements[task_id]],
                task_actor,
                request_id,
                task_id,
                placements,
                selected_mode,
                arrival_us,
            )
        if cost_model.path["kind"] == "host_staged_two_copy":
            for edge_index, edge in enumerate(dag["edges"]):
                if placements[edge["source"]] != placements[edge["target"]]:
                    Actor.create(
                        f"request-{request_id}-relay-{edge_index}",
                        hosts["HOST"],
                        relay_actor,
                        request_id,
                        edge_index,
                        edge,
                        placements,
                        selected_mode,
                    )

    for request_id in range(request_count):
        Actor.create(f"request-{request_id}-launcher", hosts["HOST"], launch_request, request_id)

    engine.run()
    completions.sort(key=lambda row: row["request_id"])
    decisions.sort(key=lambda row: row["request_id"])
    task_timeline.sort(key=lambda row: (row["start_us"], row["request_id"], row["task_id"]))
    transfer_timeline.sort(
        key=lambda row: (row["start_us"], row["request_id"], row["edge_index"], row["stage"])
    )
    if len(completions) != request_count:
        raise RuntimeError(f"Expected {request_count} completions, got {len(completions)}")

    write_csv(
        output / "request_completion.csv",
        ["request_id", "selected_mode", "arrival_us", "completion_us", "latency_us", "slo_met"],
        completions,
    )
    write_csv(
        output / "policy_decisions.csv",
        ["request_id", "arrival_us", "backlog_at_arrival", "selected_mode"],
        decisions,
    )
    write_csv(
        output / "task_timeline.csv",
        ["request_id", "selected_mode", "task_id", "task_type", "device", "start_us", "finish_us"],
        task_timeline,
    )
    write_csv(
        output / "transfer_timeline.csv",
        [
            "request_id",
            "selected_mode",
            "edge_index",
            "stage",
            "resource",
            "direction",
            "bytes",
            "start_us",
            "finish_us",
        ],
        transfer_timeline,
    )

    latencies = [row["latency_us"] for row in completions]
    first_completion_us = min(row["completion_us"] for row in completions)
    last_completion_us = max(row["completion_us"] for row in completions)
    steady_state_throughput = (
        (request_count - 1) * 1_000_000.0 / (last_completion_us - first_completion_us)
        if request_count > 1 and last_completion_us > first_completion_us
        else 1_000_000.0 / latencies[0]
    )
    mode_counts = defaultdict(int)
    for row in decisions:
        mode_counts[row["selected_mode"]] += 1
    summary = {
        "profile_kind": "sensitivity_dynamic_runtime",
        "valid_for_target_prediction": False,
        "policy": policy,
        "request_count": request_count,
        "offered_rate_requests_per_s": arrival_rate,
        "arrival_interval_us": interval_us,
        "adaptive_backlog_threshold": threshold if policy == "adaptive_dual_mode" else None,
        "slo_us": slo_us,
        "mean_latency_us": sum(latencies) / len(latencies),
        "p50_latency_us": percentile(latencies, 0.50),
        "p95_latency_us": percentile(latencies, 0.95),
        "max_latency_us": max(latencies),
        "slo_violation_rate": sum(not row["slo_met"] for row in completions) / request_count,
        "makespan_us": last_completion_us,
        "achieved_throughput_requests_per_s": steady_state_throughput,
        "finite_batch_throughput_requests_per_s": request_count * 1_000_000.0 / last_completion_us,
        "mode_counts": dict(mode_counts),
        "task_records": len(task_timeline),
        "transfer_records": len(transfer_timeline),
        "codec_resource_model": "device_compute",
        "artifacts": {
            "request_completion": "request_completion.csv",
            "policy_decisions": "policy_decisions.csv",
            "task_timeline": "task_timeline.csv",
            "transfer_timeline": "transfer_timeline.csv",
            "platform": "generated_platform.xml",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("RESULT_JSON " + json.dumps(summary, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", choices=POLICIES)
    parser.add_argument("dag", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--arrival-rate", type=float, required=True)
    parser.add_argument("--adaptive-threshold", type=int, default=2)
    parser.add_argument("--slo-us", type=float, default=30000.0)
    args = parser.parse_args()
    if args.requests < 1 or args.arrival_rate <= 0 or args.adaptive_threshold < 1:
        raise SystemExit("requests, arrival rate, and threshold must be positive")

    here = Path(__file__).resolve().parent
    runtime_path = here.parent / "12_full_dag_baselines" / "run_policy.py"
    runtime = load_runtime(runtime_path)
    dag = json.loads(args.dag.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    run_case(
        runtime,
        dag,
        profile,
        args.policy,
        args.requests,
        args.arrival_rate,
        args.adaptive_threshold,
        args.slo_us,
        args.output_directory.resolve(),
    )


if __name__ == "__main__":
    main()
