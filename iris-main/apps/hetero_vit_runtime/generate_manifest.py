import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path


def load_runtime(path):
    spec = importlib.util.spec_from_file_location("full_dag_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def c_string(value):
    return json.dumps(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dag", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output_header", type=Path)
    parser.add_argument("--objective", choices=("latency", "throughput"), default="latency")
    parser.add_argument("--placements", type=Path,
                        help="Optional fixed DP task-device map; absent means policy chooses devices")
    parser.add_argument("--output-json", type=Path,
                        help="Optional machine-readable copy of the generated task and edge manifest")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    runtime_path = here.parents[2] / "simgrid_baseline" / "12_full_dag_baselines" / "run_policy.py"
    runtime = load_runtime(runtime_path)
    dag = json.loads(args.dag.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    cost_model = runtime.SensitivityCostModel(profile)
    order = runtime.topological_order(dag["tasks"], dag["edges"])
    task_by_id = {task["id"]: task for task in dag["tasks"]}
    placements = (json.loads(args.placements.read_text(encoding="utf-8"))
                  if args.placements is not None else None)
    if placements is not None:
        if set(placements) != set(task_by_id):
            raise RuntimeError("Placement map must contain exactly one device per DAG task")
        for task_id, device in placements.items():
            if device not in task_by_id[task_id]["candidate_devices"]:
                raise RuntimeError(f"Unsupported placement: {task_id} on {device}")
    index_by_id = {task_id: index for index, task_id in enumerate(order)}
    incoming_bytes = defaultdict(int)
    outgoing_bytes = defaultdict(int)
    gpu_only_successor = defaultdict(bool)
    for edge in dag["edges"]:
        encoded = cost_model.edge_bytes(edge)
        incoming_bytes[edge["target"]] += encoded
        outgoing_bytes[edge["source"]] += encoded
        if task_by_id[edge["target"]]["candidate_devices"] == ["GPU"]:
            gpu_only_successor[edge["source"]] = True

    terminal_tasks = {task["id"] for task in dag["tasks"]} - set(outgoing_bytes)
    for task_id in terminal_tasks:
        task = task_by_id[task_id]
        outgoing_bytes[task_id] = (
            runtime.element_count(task["output_shape_batch1"]) * cost_model.bytes_per_element
        )
        gpu_only_successor[task_id] = True

    objective = 0 if args.objective == "latency" else 1
    task_rows = []
    for task_id in order:
        task = task_by_id[task_id]
        eligible = "FPGA" in task["candidate_devices"]
        planned = 0 if placements is None else (1 if placements[task_id] == "GPU" else 2)
        gpu_us = int(round(cost_model.latency_us(task, "GPU")))
        fpga_us = int(round(cost_model.latency_us(task, "FPGA"))) if eligible else 0
        task_rows.append(
            (
                task_id,
                gpu_us,
                fpga_us,
                incoming_bytes[task_id],
                outgoing_bytes[task_id],
                int(eligible),
                objective,
                int(gpu_only_successor[task_id]),
                cost_model.linear_equivalent_count(task),
                planned,
            )
        )

    edge_rows = [(index_by_id[e["source"]], index_by_id[e["target"]]) for e in dag["edges"]]
    lines = [
        "#ifndef IRIS_APPS_HETERO_VIT_TASK_MANIFEST_H",
        "#define IRIS_APPS_HETERO_VIT_TASK_MANIFEST_H",
        "",
        "typedef struct {",
        "  const char* name;",
        "  int metadata[9];",
        "} HeteroTaskRecord;",
        "",
        "typedef struct { int source; int target; } HeteroEdgeRecord;",
        "",
        f"static const int HETERO_TASK_COUNT = {len(task_rows)};",
        f"static const int HETERO_EDGE_COUNT = {len(edge_rows)};",
        "static const HeteroTaskRecord HETERO_TASKS[] = {",
    ]
    for row in task_rows:
        lines.append(
            "  {%s, {%d, %d, %d, %d, %d, %d, %d, %d, %d}},"
            % (c_string(row[0]), *row[1:])
        )
    lines.extend(["};", "", "static const HeteroEdgeRecord HETERO_EDGES[] = {"])
    for source, target in edge_rows:
        lines.append(f"  {{{source}, {target}}},")
    lines.extend(["};", "", "#endif", ""])
    args.output_header.write_text("\n".join(lines), encoding="ascii")
    if args.output_json is not None:
        manifest = {
            "status": "deployment_manifest_not_hardware_validation",
            "dag_task_count": len(task_rows),
            "dag_edge_count": len(edge_rows),
            "tasks": [
                {"id": row[0], "metadata": list(row[1:]),
                 "planned_device": ("AUTO", "GPU", "FPGA")[row[9]]}
                for row in task_rows
            ],
            "edges": [{"source_index": source, "target_index": target}
                      for source, target in edge_rows],
        }
        args.output_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "tasks": len(task_rows),
                "edges": len(edge_rows),
                "fpga_eligible_tasks": sum(row[5] for row in task_rows),
                "linear_equivalents": sum(row[8] for row in task_rows),
                "objective": args.objective,
                "planned_fpga_tasks": sum(row[9] == 2 for row in task_rows),
                "output": str(args.output_header.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
