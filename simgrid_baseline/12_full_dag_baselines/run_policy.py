import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path

try:
    from simgrid import Actor, Engine, Mailbox, Mutex, Semaphore, this_actor
except ModuleNotFoundError:
    Actor = Engine = Mailbox = Mutex = Semaphore = this_actor = None


POLICIES = (
    "gpu_only",
    "static_all_fpga",
    "iris_profile_compute_only",
    "iris_data_locality",
    "communication_eft",
)
LINEAR_TYPES = {
    "linear_qkv",
    "linear_projection",
    "linear_ffn1",
    "linear_ffn2",
    "linear_classifier",
}
FUSED_FFN_TYPE = "fused_ffn_island"
PATH_KINDS = {
    "host_staged_two_copy",
    "shared_mapped_one_copy",
    "direct_dma_one_copy",
}


def element_count(shape):
    return math.prod(shape)


def topological_order(tasks, edges):
    task_ids = [task["id"] for task in tasks]
    indegree = {task_id: 0 for task_id in task_ids}
    outgoing = defaultdict(list)
    for index, edge in enumerate(edges):
        outgoing[edge["source"]].append(index)
        indegree[edge["target"]] += 1
    ready = deque(task_id for task_id in task_ids if indegree[task_id] == 0)
    order = []
    while ready:
        task_id = ready.popleft()
        order.append(task_id)
        for edge_index in outgoing[task_id]:
            target = edges[edge_index]["target"]
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(order) != len(tasks):
        raise RuntimeError("Task graph is cyclic")
    return order


class SensitivityCostModel:
    def __init__(self, profile):
        profile_kind = profile.get("profile_kind")
        if profile_kind == "sensitivity":
            if profile.get("valid_for_target_prediction") is not False:
                raise RuntimeError("Sensitivity profile must declare valid_for_target_prediction=false")
        elif profile_kind == "target_measurement":
            if profile.get("status") != "complete":
                raise RuntimeError("Target measurement profile is incomplete")
            if profile.get("valid_for_target_prediction") is not True:
                raise RuntimeError("Complete target profile must declare valid_for_target_prediction=true")
            if not profile.get("task_costs_us"):
                raise RuntimeError("Complete target profile has no measured task costs")
        else:
            raise RuntimeError(f"Unsupported profile_kind={profile_kind!r}")
        self.profile_kind = profile_kind
        self.profile = profile
        self.bytes_per_element = profile["activation"]["bytes_per_element"]
        self.gpu = profile.get("gpu", {})
        self.fpga = profile.get("fpga", {})
        self.task_costs_us = profile.get("task_costs_us", {})
        self.path = profile["communication_path"]
        self.compression = profile.get(
            "activation_compression",
            {
                "ratio": 1.0,
                "encode_fixed_us": 0.0,
                "decode_fixed_us": 0.0,
                "encode_bandwidth_GBps": 0.0,
                "decode_bandwidth_GBps": 0.0,
            },
        )
        self.fpga_input_buffers = profile.get("pipeline", {}).get("fpga_input_buffers")
        if self.path["kind"] not in PATH_KINDS:
            raise RuntimeError(f"Unsupported communication path {self.path['kind']}")
        ratio = self.compression["ratio"]
        if not 0.0 < ratio <= 1.0:
            raise RuntimeError("activation_compression.ratio must be in (0, 1]")
        if self.fpga_input_buffers is not None and self.fpga_input_buffers < 1:
            raise RuntimeError("pipeline.fpga_input_buffers must be at least one")

    @staticmethod
    def time_from_flops(flops, effective_tflops):
        return flops / (effective_tflops * 1_000_000.0)

    @staticmethod
    def time_from_bytes(byte_count, bandwidth_gbps):
        return byte_count / (bandwidth_gbps * 1000.0)

    def linear_flops(self, task):
        input_shape = task["input_shape_batch1"]
        output_shape = task["output_shape_batch1"]
        vectors = element_count(input_shape[:-1])
        return 2 * vectors * input_shape[-1] * output_shape[-1]

    def measured_latency_us(self, task, device):
        value = self.task_costs_us.get(task["id"], {}).get(device)
        return float(value) if value is not None else None

    def gpu_latency_us(self, task):
        measured = self.measured_latency_us(task, "GPU")
        if measured is not None:
            return measured
        if self.profile_kind == "target_measurement":
            raise RuntimeError(f"Missing measured GPU cost for task {task['id']}")
        task_type = task["task_type"]
        if not self.gpu:
            raise RuntimeError(f"Missing measured GPU cost for task {task['id']}")
        launch = self.gpu["launch_overhead_us"]
        output_bytes = element_count(task["output_shape_batch1"]) * self.bytes_per_element

        if task_type == FUSED_FFN_TYPE:
            return sum(self.gpu_latency_us(component) for component in task["fused_components"])
        if task_type in LINEAR_TYPES:
            return launch + self.time_from_flops(
                self.linear_flops(task), self.gpu["matmul_effective_tflops"]
            )
        if task_type == "patch_embedding":
            patches = task["output_shape_batch1"][1]
            input_features = 16 * 16 * 3
            output_features = task["output_shape_batch1"][-1]
            flops = 2 * patches * input_features * output_features
            return launch + self.time_from_flops(flops, self.gpu["matmul_effective_tflops"])
        if task_type == "self_attention":
            tokens = task["output_shape_batch1"][1]
            hidden = task["output_shape_batch1"][2]
            flops = 4 * tokens * tokens * hidden
            return launch + self.time_from_flops(flops, self.gpu["attention_effective_tflops"])

        traffic_multipliers = {
            "token_and_position_embedding": 3,
            "layer_norm": 4,
            "residual_add": 3,
            "gelu": 2,
            "class_token_select": 2,
        }
        if task_type not in traffic_multipliers:
            raise RuntimeError(f"No sensitivity cost rule for task type {task_type}")
        traffic_bytes = traffic_multipliers[task_type] * output_bytes
        return launch + self.time_from_bytes(traffic_bytes, self.gpu["memory_bandwidth_GBps"])

    def latency_us(self, task, device):
        measured = self.measured_latency_us(task, device)
        if measured is not None:
            return measured
        if self.profile_kind == "target_measurement":
            raise RuntimeError(f"Missing measured {device} cost for task {task['id']}")
        gpu_latency = self.gpu_latency_us(task)
        if device == "GPU":
            return gpu_latency
        if device != "FPGA" or task["task_type"] not in LINEAR_TYPES | {FUSED_FFN_TYPE}:
            raise RuntimeError(f"Unsupported placement: {task['id']} on {device}")
        if not self.fpga:
            raise RuntimeError(f"Missing measured FPGA cost for task {task['id']}")
        if task["task_type"] == FUSED_FFN_TYPE:
            launch = self.gpu["launch_overhead_us"]
            linear_compute = 0.0
            elementwise_compute = 0.0
            for component in task["fused_components"]:
                compute = max(0.0, self.gpu_latency_us(component) - launch)
                if component["task_type"] in LINEAR_TYPES:
                    linear_compute += compute
                else:
                    elementwise_compute += compute
            elementwise_speedup = self.fpga.get("elementwise_speedup_over_gpu", 1.0)
            return (
                linear_compute / self.fpga["linear_speedup_over_gpu"]
                + elementwise_compute / elementwise_speedup
                + self.fpga["dispatch_overhead_us"]
            )
        return gpu_latency / self.fpga["linear_speedup_over_gpu"] + self.fpga["dispatch_overhead_us"]

    def uncompressed_edge_bytes(self, edge):
        fp32_bytes = edge["bytes_batch1_fp32"]
        return int(round(fp32_bytes * self.bytes_per_element / 4.0))

    def edge_bytes(self, edge):
        return int(round(self.uncompressed_edge_bytes(edge) * self.compression["ratio"]))

    def codec_latency_us(self, edge, operation):
        fixed = self.compression[f"{operation}_fixed_us"]
        bandwidth = self.compression[f"{operation}_bandwidth_GBps"]
        variable = 0.0
        if bandwidth > 0.0:
            variable = self.time_from_bytes(self.uncompressed_edge_bytes(edge), bandwidth)
        return fixed + variable

    @staticmethod
    def linear_equivalent_count(task):
        if task["task_type"] in LINEAR_TYPES:
            return 1
        if task["task_type"] == FUSED_FFN_TYPE:
            return sum(
                component["task_type"] in LINEAR_TYPES
                for component in task["fused_components"]
            )
        return 0

    def stage_latency_us(self, stage, byte_count):
        values = self.path[stage]
        return values["fixed_latency_us"] + self.time_from_bytes(
            byte_count, values["effective_bandwidth_GBps"]
        )

    def link_resources(self):
        if self.path["kind"] == "host_staged_two_copy":
            return ("gpu_host", "host_fpga_pcie")
        if self.path["kind"] == "shared_mapped_one_copy":
            return ("host_fpga_pcie",)
        return ("direct_gpu_fpga",)

    def transfer_stages(self, source_device):
        kind = self.path["kind"]
        if kind == "host_staged_two_copy":
            if source_device == "GPU":
                return (
                    ("gpu_host", "GPU_to_HOST"),
                    ("host_fpga_pcie", "HOST_to_FPGA"),
                )
            return (
                ("host_fpga_pcie", "FPGA_to_HOST"),
                ("gpu_host", "HOST_to_GPU"),
            )
        if kind == "shared_mapped_one_copy":
            direction = "SHARED_to_FPGA" if source_device == "GPU" else "FPGA_to_SHARED"
            return (("host_fpga_pcie", direction),)
        direction = "GPU_to_FPGA" if source_device == "GPU" else "FPGA_to_GPU"
        return (("direct_gpu_fpga", direction),)


def graph_indexes(dag):
    tasks = {task["id"]: task for task in dag["tasks"]}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for index, edge in enumerate(dag["edges"]):
        incoming[edge["target"]].append((index, edge))
        outgoing[edge["source"]].append((index, edge))
    return tasks, incoming, outgoing


def initial_placements(policy, order, tasks, incoming, cost_model):
    placements = {}
    for task_id in order:
        task = tasks[task_id]
        candidates = task["candidate_devices"]
        if policy == "gpu_only":
            placements[task_id] = "GPU"
        elif policy == "static_all_fpga":
            placements[task_id] = "FPGA" if "FPGA" in candidates else "GPU"
        elif policy == "iris_profile_compute_only":
            placements[task_id] = min(
                candidates,
                key=lambda device: (cost_model.latency_us(task, device), device != "GPU"),
            )
        elif policy == "iris_data_locality":
            resident_bytes = {device: 0 for device in candidates}
            for _edge_index, edge in incoming[task_id]:
                source_device = placements[edge["source"]]
                if source_device in resident_bytes:
                    resident_bytes[source_device] += cost_model.edge_bytes(edge)
            placements[task_id] = max(
                candidates,
                key=lambda device: (resident_bytes[device], device == "GPU"),
            )
        else:
            raise RuntimeError(f"Initial placement is not defined for {policy}")
    return placements


def transfer_plan(source_device, target_device, ready_us, edge, link_available, cost_model):
    if source_device == target_device:
        return ready_us, dict(link_available), []
    if {source_device, target_device} != {"GPU", "FPGA"}:
        raise RuntimeError(f"Unsupported transfer {source_device}->{target_device}")

    available = dict(link_available)
    byte_count = cost_model.edge_bytes(edge)
    current = ready_us + cost_model.codec_latency_us(edge, "encode")
    rows = []
    for resource, direction in cost_model.transfer_stages(source_device):
        start = max(current, available[resource])
        finish = start + cost_model.stage_latency_us(resource, byte_count)
        available[resource] = finish
        rows.append(
            {
                "resource": resource,
                "direction": direction,
                "start_us": start,
                "finish_us": finish,
                "bytes": byte_count,
            }
        )
        current = finish
    current += cost_model.codec_latency_us(edge, "decode")
    return current, available, rows


def evaluate_candidate(
    task_id,
    device,
    tasks,
    incoming,
    outgoing,
    placements,
    finish_times,
    device_available,
    link_available,
    cost_model,
):
    candidate_links = dict(link_available)
    arrivals = []
    transfers = []
    sorted_inputs = sorted(incoming[task_id], key=lambda item: finish_times[item[1]["source"]])
    for edge_index, edge in sorted_inputs:
        source = edge["source"]
        arrival, candidate_links, stages = transfer_plan(
            placements[source],
            device,
            finish_times[source],
            edge,
            candidate_links,
            cost_model,
        )
        arrivals.append(arrival)
        for stage in stages:
            stage.update({"edge_index": edge_index, "source": source, "target": task_id})
            transfers.append(stage)
    ready = max(arrivals, default=0.0)
    start = max(ready, device_available[device])
    finish = start + cost_model.latency_us(tasks[task_id], device)

    # A Linear placed on FPGA is normally followed immediately by a mandatory
    # GPU task. Include that known return path in the placement decision, while
    # leaving the actual transfer reservation to the successor scheduling step.
    decision_finish = finish
    lookahead_links = dict(candidate_links)
    for _edge_index, edge in outgoing[task_id]:
        target_devices = tasks[edge["target"]]["candidate_devices"]
        if device == "FPGA" and target_devices == ["GPU"]:
            lookahead_finish, lookahead_links, _stages = transfer_plan(
                "FPGA",
                "GPU",
                finish,
                edge,
                lookahead_links,
                cost_model,
            )
            decision_finish = max(decision_finish, lookahead_finish)
    return {
        "device": device,
        "start_us": start,
        "finish_us": finish,
        "decision_finish_us": decision_finish,
        "link_available": candidate_links,
        "transfers": transfers,
    }


def analytical_schedule(policy, order, tasks, incoming, outgoing, cost_model):
    placements = {}
    finish_times = {}
    device_available = {"GPU": 0.0, "FPGA": 0.0}
    link_available = {resource: 0.0 for resource in cost_model.link_resources()}
    transfer_rows = []
    task_rows = []

    fixed = None
    if policy != "communication_eft":
        fixed = initial_placements(policy, order, tasks, incoming, cost_model)

    for task_id in order:
        task = tasks[task_id]
        candidates = task["candidate_devices"] if fixed is None else [fixed[task_id]]
        options = [
            evaluate_candidate(
                task_id,
                device,
                tasks,
                incoming,
                outgoing,
                placements,
                finish_times,
                device_available,
                link_available,
                cost_model,
            )
            for device in candidates
        ]
        selected = min(
            options,
            key=lambda option: (option["decision_finish_us"], option["device"] != "GPU"),
        )
        device = selected["device"]
        placements[task_id] = device
        finish_times[task_id] = selected["finish_us"]
        device_available[device] = selected["finish_us"]
        link_available = selected["link_available"]
        transfer_rows.extend(selected["transfers"])
        task_rows.append(
            {
                "task_id": task_id,
                "task_type": task["task_type"],
                "device": device,
                "start_us": selected["start_us"],
                "finish_us": selected["finish_us"],
                "duration_us": cost_model.latency_us(task, device),
            }
        )

    return {
        "placements": placements,
        "task_rows": task_rows,
        "transfer_rows": transfer_rows,
        "makespan_us": max(finish_times.values()),
    }


def write_platform(profile, path):
    communication = profile["communication_path"]
    kind = communication["kind"]
    if kind == "host_staged_two_copy":
        gpu_host = communication["gpu_host"]
        pcie = communication["host_fpga_pcie"]
        links_and_routes = f'''
    <link id="gpu_host" bandwidth="{gpu_host['effective_bandwidth_GBps']}GBps" latency="{gpu_host['fixed_latency_us']}us" sharing_policy="SHARED"/>
    <link id="host_fpga_pcie" bandwidth="{pcie['effective_bandwidth_GBps']}GBps" latency="{pcie['fixed_latency_us']}us" sharing_policy="SHARED"/>
    <route src="GPU" dst="HOST"><link_ctn id="gpu_host"/></route>
    <route src="HOST" dst="FPGA"><link_ctn id="host_fpga_pcie"/></route>'''
    else:
        resource = "host_fpga_pcie" if kind == "shared_mapped_one_copy" else "direct_gpu_fpga"
        link = communication[resource]
        links_and_routes = f'''
    <link id="{resource}" bandwidth="{link['effective_bandwidth_GBps']}GBps" latency="{link['fixed_latency_us']}us" sharing_policy="SHARED"/>
    <route src="GPU" dst="FPGA"><link_ctn id="{resource}"/></route>'''

    text = f'''<?xml version="1.0"?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="jetson_host_fpga" routing="Full">
    <host id="GPU" speed="1Gf"/>
    <host id="HOST" speed="1Gf"/>
    <host id="FPGA" speed="1Gf"/>
{links_and_routes}
  </zone>
</platform>
'''
    path.write_text(text, encoding="ascii")


def run_simgrid(dag, cost_model, placements, platform_path, request_count=1):
    if Engine is None:
        raise RuntimeError(
            "Python SimGrid bindings are not installed. Analytical scheduling APIs "
            "can still be imported, but SimGrid replay requires installing simgrid."
        )
    tasks, incoming, outgoing = graph_indexes(dag)
    engine = Engine(["run_policy.py"])
    Engine.set_config("network/model:CM02")
    engine.load_platform(str(platform_path))
    hosts = {name: engine.host_by_name(name) for name in ("GPU", "HOST", "FPGA")}
    locks = {"GPU": Mutex(), "FPGA": Mutex()}
    fpga_buffers = (
        Semaphore(cost_model.fpga_input_buffers)
        if cost_model.fpga_input_buffers is not None
        else None
    )
    task_timeline = []
    transfer_timeline = []
    request_completion_us = []

    def target_mailbox(request_id, edge_index):
        return Mailbox.by_name(f"request-{request_id}-edge-{edge_index}-target")

    def relay_mailbox(request_id, edge_index):
        return Mailbox.by_name(f"request-{request_id}-edge-{edge_index}-relay")

    def task_actor(request_id, task_id):
        task = tasks[task_id]
        device = placements[task_id]
        cross_inputs = []
        for edge_index, edge in incoming[task_id]:
            target_mailbox(request_id, edge_index).get()
            if placements[edge["source"]] != device:
                cross_inputs.append(edge)
                codec_us = cost_model.codec_latency_us(edge, "decode")
                if codec_us:
                    this_actor.sleep_for(codec_us / 1_000_000.0)

        locks[device].lock()
        start = engine.clock
        this_actor.execute(cost_model.latency_us(task, device) * 1000.0)
        finish = engine.clock
        locks[device].unlock()
        if fpga_buffers is not None and device == "FPGA" and cross_inputs:
            fpga_buffers.release()
        task_timeline.append(
            {
                "request_id": request_id,
                "task_id": task_id,
                "task_type": task["task_type"],
                "device": device,
                "start_us": start * 1_000_000.0,
                "finish_us": finish * 1_000_000.0,
                "duration_us": (finish - start) * 1_000_000.0,
            }
        )
        if not outgoing[task_id]:
            request_completion_us.append(
                {"request_id": request_id, "completion_us": finish * 1_000_000.0}
            )

        for edge_index, edge in outgoing[task_id]:
            target_device = placements[edge["target"]]
            if target_device == device:
                target_mailbox(request_id, edge_index).put(edge_index, 0)
                continue
            if fpga_buffers is not None and device == "GPU" and target_device == "FPGA":
                fpga_buffers.acquire()
            codec_us = cost_model.codec_latency_us(edge, "encode")
            if codec_us:
                this_actor.sleep_for(codec_us / 1_000_000.0)
            byte_count = cost_model.edge_bytes(edge)
            stage_start = engine.clock
            if cost_model.path["kind"] == "host_staged_two_copy":
                relay_mailbox(request_id, edge_index).put(edge_index, byte_count)
            else:
                target_mailbox(request_id, edge_index).put(edge_index, byte_count)
            stage_finish = engine.clock
            resource, direction = cost_model.transfer_stages(device)[0]
            transfer_timeline.append(
                {
                    "request_id": request_id,
                    "edge_index": edge_index,
                    "source": task_id,
                    "target": edge["target"],
                    "stage": 1,
                    "resource": resource,
                    "direction": direction,
                    "bytes": byte_count,
                    "start_us": stage_start * 1_000_000.0,
                    "finish_us": stage_finish * 1_000_000.0,
                    "duration_us": (stage_finish - stage_start) * 1_000_000.0,
                }
            )

    def relay_actor(request_id, edge_index, edge):
        relay_mailbox(request_id, edge_index).get()
        source_device = placements[edge["source"]]
        byte_count = cost_model.edge_bytes(edge)
        stage_start = engine.clock
        target_mailbox(request_id, edge_index).put(edge_index, byte_count)
        stage_finish = engine.clock
        resource, direction = cost_model.transfer_stages(source_device)[1]
        transfer_timeline.append(
            {
                "request_id": request_id,
                "edge_index": edge_index,
                "source": edge["source"],
                "target": edge["target"],
                "stage": 2,
                "resource": resource,
                "direction": direction,
                "bytes": byte_count,
                "start_us": stage_start * 1_000_000.0,
                "finish_us": stage_finish * 1_000_000.0,
                "duration_us": (stage_finish - stage_start) * 1_000_000.0,
            }
        )

    if fpga_buffers is not None:
        for task_id, task_inputs in incoming.items():
            if placements[task_id] != "FPGA":
                continue
            cross_input_count = sum(
                placements[edge["source"]] == "GPU" for _edge_index, edge in task_inputs
            )
            if cross_input_count > 1:
                raise RuntimeError(
                    f"Buffered replay supports one GPU input edge per FPGA task, got "
                    f"{cross_input_count} for {task_id}"
                )

    for request_id in range(request_count):
        for task_id in tasks:
            Actor.create(
                f"request-{request_id}-task-{task_id}",
                hosts[placements[task_id]],
                task_actor,
                request_id,
                task_id,
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
                    )

    engine.run()
    task_timeline.sort(key=lambda row: (row["start_us"], row["request_id"], row["task_id"]))
    transfer_timeline.sort(
        key=lambda row: (row["start_us"], row["request_id"], row["edge_index"], row["stage"])
    )
    request_completion_us.sort(key=lambda row: row["request_id"])
    return engine.clock * 1_000_000.0, task_timeline, transfer_timeline, request_completion_us


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", choices=POLICIES)
    parser.add_argument("dag", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--requests", type=int, default=1)
    args = parser.parse_args()
    if args.requests < 1:
        raise SystemExit("--requests must be at least one")

    dag = json.loads(args.dag.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    cost_model = SensitivityCostModel(profile)
    tasks, incoming, outgoing = graph_indexes(dag)
    order = topological_order(dag["tasks"], dag["edges"])
    analytical = analytical_schedule(args.policy, order, tasks, incoming, outgoing, cost_model)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    platform_path = args.output_directory / "generated_platform.xml"
    write_platform(profile, platform_path)
    makespan_us, task_timeline, transfer_timeline, request_completion = run_simgrid(
        dag,
        cost_model,
        analytical["placements"],
        platform_path,
        request_count=args.requests,
    )

    placement_rows = [
        {
            "task_id": task_id,
            "task_type": tasks[task_id]["task_type"],
            "device": analytical["placements"][task_id],
            "modeled_compute_us": f"{cost_model.latency_us(tasks[task_id], analytical['placements'][task_id]):.6f}",
        }
        for task_id in order
    ]
    write_csv(
        args.output_directory / "placements.csv",
        ["task_id", "task_type", "device", "modeled_compute_us"],
        placement_rows,
    )
    write_csv(
        args.output_directory / "task_timeline.csv",
        ["request_id", "task_id", "task_type", "device", "start_us", "finish_us", "duration_us"],
        task_timeline,
    )
    write_csv(
        args.output_directory / "transfer_timeline.csv",
        [
            "request_id",
            "edge_index",
            "source",
            "target",
            "stage",
            "resource",
            "direction",
            "bytes",
            "start_us",
            "finish_us",
            "duration_us",
        ],
        transfer_timeline,
    )
    write_csv(
        args.output_directory / "request_completion.csv",
        ["request_id", "completion_us"],
        request_completion,
    )

    cross_edges = [
        edge
        for edge in dag["edges"]
        if analytical["placements"][edge["source"]] != analytical["placements"][edge["target"]]
    ]
    uncompressed_bytes = sum(cost_model.uncompressed_edge_bytes(edge) for edge in cross_edges)
    logical_bytes = sum(cost_model.edge_bytes(edge) for edge in cross_edges)
    communication_stages = len(cost_model.transfer_stages("GPU"))
    offloaded = sum(
        cost_model.linear_equivalent_count(task)
        if analytical["placements"][task["id"]] == "FPGA"
        else 0
        for task in dag["tasks"]
    )
    codec_us = sum(
        cost_model.codec_latency_us(edge, "encode")
        + cost_model.codec_latency_us(edge, "decode")
        for edge in cross_edges
    )
    summary = {
        "policy": args.policy,
        "profile_kind": profile["profile_kind"],
        "profile_name": profile["profile_name"],
        "valid_for_target_prediction": profile["valid_for_target_prediction"],
        "communication_path_kind": cost_model.path["kind"],
        "activation_compression_ratio": cost_model.compression["ratio"],
        "fpga_input_buffers": cost_model.fpga_input_buffers,
        "request_count": args.requests,
        "tasks_per_request": len(dag["tasks"]),
        "task_records": len(task_timeline),
        "request_completion_records": len(request_completion),
        "edges": len(dag["edges"]),
        "offloaded_linear_tasks": offloaded,
        "cross_device_edges": len(cross_edges),
        "uncompressed_cross_device_bytes": uncompressed_bytes,
        "logical_cross_device_bytes": logical_bytes,
        "communication_stages_per_cross_edge": communication_stages,
        "transfer_records": len(transfer_timeline),
        "physical_link_bytes": logical_bytes * communication_stages,
        "total_logical_cross_device_bytes": logical_bytes * args.requests,
        "total_physical_link_bytes": logical_bytes * communication_stages * args.requests,
        "host_staged_link_bytes": logical_bytes * communication_stages,
        "endpoint_codec_us_per_request": codec_us,
        "gpu_compute_busy_us": sum(
            cost_model.latency_us(tasks[task_id], "GPU")
            for task_id, device in analytical["placements"].items()
            if device == "GPU"
        ),
        "fpga_compute_busy_us": sum(
            cost_model.latency_us(tasks[task_id], "FPGA")
            for task_id, device in analytical["placements"].items()
            if device == "FPGA"
        ),
        "simgrid_makespan_us": makespan_us,
        "throughput_requests_per_s": args.requests * 1_000_000.0 / makespan_us,
        "mean_completion_us": sum(row["completion_us"] for row in request_completion) / args.requests,
        "single_request_analytical_scheduler_makespan_us": analytical["makespan_us"],
        "simgrid_minus_analytical_us": (
            makespan_us - analytical["makespan_us"] if args.requests == 1 else None
        ),
        "artifacts": {
            "placements": "placements.csv",
            "task_timeline": "task_timeline.csv",
            "transfer_timeline": "transfer_timeline.csv",
            "request_completion": "request_completion.csv",
            "platform": "generated_platform.xml"
        },
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("RESULT_JSON " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
