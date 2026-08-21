import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path


BATCH = 1
TOKENS = 197
PATCH_TOKENS = 196
HIDDEN = 768
MLP_HIDDEN = 3072
CLASSES = 1000
FP32_BYTES = 4


def tensor_bytes(*shape):
    result = FP32_BYTES
    for dimension in shape:
        result *= dimension
    return result


TOKEN_BYTES = tensor_bytes(BATCH, TOKENS, HIDDEN)
PATCH_TOKEN_BYTES = tensor_bytes(BATCH, PATCH_TOKENS, HIDDEN)
QKV_BYTES = tensor_bytes(BATCH, TOKENS, HIDDEN * 3)
MLP_BYTES = tensor_bytes(BATCH, TOKENS, MLP_HIDDEN)
CLASS_TOKEN_BYTES = tensor_bytes(BATCH, HIDDEN)


def group_for_raw_node(node):
    if not node["runtime_task"] or node["op_type"] == "Identity":
        return None

    name = node["name"]
    match = re.match(r"/blocks/blocks\.(\d+)/(.*)", name)
    if match:
        block = int(match.group(1))
        suffix = match.group(2)
        prefix = f"block_{block:02d}"
        if suffix.startswith("norm1/"):
            return f"{prefix}_norm1"
        if suffix.startswith("attn/qkv/"):
            return f"{prefix}_qkv"
        if suffix.startswith("attn/proj/"):
            return f"{prefix}_proj"
        if suffix.startswith("attn/"):
            return f"{prefix}_attention"
        if suffix == "Add":
            return f"{prefix}_residual1"
        if suffix.startswith("norm2/"):
            return f"{prefix}_norm2"
        if suffix.startswith("mlp/fc1/"):
            return f"{prefix}_fc1"
        if suffix.startswith("mlp/act/"):
            return f"{prefix}_gelu"
        if suffix.startswith("mlp/fc2/"):
            return f"{prefix}_fc2"
        if suffix == "Add_1":
            return f"{prefix}_residual2"
        raise RuntimeError(f"Unclassified block node: {name}")

    if node["index"] < 124:
        return "patch_embedding" if name.startswith("/patch_embed/") else "token_embedding"
    if name.startswith("/norm/"):
        return "final_norm"
    if name == "/Gather_1":
        return "class_token_select"
    if name.startswith("/head/"):
        return "classifier"
    raise RuntimeError(f"Unclassified runtime node: {name}")


def add_task(tasks, task_id, task_type, devices, input_shape, output_shape, raw_groups):
    tasks.append(
        {
            "id": task_id,
            "task_type": task_type,
            "candidate_devices": devices,
            "input_shape_batch1": input_shape,
            "output_shape_batch1": output_shape,
            "source_onnx_node_ids": raw_groups.get(task_id, []),
        }
    )


def add_edge(edges, source, target, role, byte_count):
    edges.append(
        {
            "source": source,
            "target": target,
            "tensor_role": role,
            "bytes_batch1_fp32": byte_count,
        }
    )


def validate_dag(tasks, edges):
    task_ids = {task["id"] for task in tasks}
    if len(task_ids) != len(tasks):
        raise RuntimeError("Duplicate coarse task IDs")
    for edge in edges:
        if edge["source"] not in task_ids or edge["target"] not in task_ids:
            raise RuntimeError(f"Edge references an unknown task: {edge}")

    successors = defaultdict(list)
    indegree = {task_id: 0 for task_id in task_ids}
    for edge in edges:
        successors[edge["source"]].append(edge["target"])
        indegree[edge["target"]] += 1
    ready = deque(task_id for task_id, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        source = ready.popleft()
        visited += 1
        for target in successors[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited == len(tasks)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: python {sys.argv[0]} <raw-dag.json> <output-directory>")

    raw_dag_path = Path(sys.argv[1]).resolve()
    output_directory = Path(sys.argv[2]).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_dag = json.loads(raw_dag_path.read_text(encoding="utf-8"))

    raw_groups = defaultdict(list)
    ignored_identity_nodes = 0
    for node in raw_dag["nodes"]:
        group = group_for_raw_node(node)
        if group is not None:
            raw_groups[group].append(node["id"])
        elif node["op_type"] == "Identity":
            ignored_identity_nodes += 1

    tasks = []
    edges = []
    add_task(tasks, "patch_embedding", "patch_embedding", ["GPU"], [1, 3, 224, 224], [1, PATCH_TOKENS, HIDDEN], raw_groups)
    add_task(tasks, "token_embedding", "token_and_position_embedding", ["GPU"], [1, PATCH_TOKENS, HIDDEN], [1, TOKENS, HIDDEN], raw_groups)
    add_edge(edges, "patch_embedding", "token_embedding", "patch_tokens", PATCH_TOKEN_BYTES)

    previous = "token_embedding"
    for block in range(12):
        prefix = f"block_{block:02d}"
        norm1 = f"{prefix}_norm1"
        qkv = f"{prefix}_qkv"
        attention = f"{prefix}_attention"
        proj = f"{prefix}_proj"
        residual1 = f"{prefix}_residual1"
        norm2 = f"{prefix}_norm2"
        fc1 = f"{prefix}_fc1"
        gelu = f"{prefix}_gelu"
        fc2 = f"{prefix}_fc2"
        residual2 = f"{prefix}_residual2"

        add_task(tasks, norm1, "layer_norm", ["GPU"], [1, TOKENS, HIDDEN], [1, TOKENS, HIDDEN], raw_groups)
        add_task(tasks, qkv, "linear_qkv", ["GPU", "FPGA"], [1, TOKENS, HIDDEN], [1, TOKENS, HIDDEN * 3], raw_groups)
        add_task(tasks, attention, "self_attention", ["GPU"], [1, TOKENS, HIDDEN * 3], [1, TOKENS, HIDDEN], raw_groups)
        add_task(tasks, proj, "linear_projection", ["GPU", "FPGA"], [1, TOKENS, HIDDEN], [1, TOKENS, HIDDEN], raw_groups)
        add_task(tasks, residual1, "residual_add", ["GPU"], [1, TOKENS, HIDDEN], [1, TOKENS, HIDDEN], raw_groups)
        add_task(tasks, norm2, "layer_norm", ["GPU"], [1, TOKENS, HIDDEN], [1, TOKENS, HIDDEN], raw_groups)
        add_task(tasks, fc1, "linear_ffn1", ["GPU", "FPGA"], [1, TOKENS, HIDDEN], [1, TOKENS, MLP_HIDDEN], raw_groups)
        add_task(tasks, gelu, "gelu", ["GPU"], [1, TOKENS, MLP_HIDDEN], [1, TOKENS, MLP_HIDDEN], raw_groups)
        add_task(tasks, fc2, "linear_ffn2", ["GPU", "FPGA"], [1, TOKENS, MLP_HIDDEN], [1, TOKENS, HIDDEN], raw_groups)
        add_task(tasks, residual2, "residual_add", ["GPU"], [1, TOKENS, HIDDEN], [1, TOKENS, HIDDEN], raw_groups)

        add_edge(edges, previous, norm1, "block_input", TOKEN_BYTES)
        add_edge(edges, previous, residual1, "residual_skip_1", TOKEN_BYTES)
        add_edge(edges, norm1, qkv, "normalized_tokens", TOKEN_BYTES)
        add_edge(edges, qkv, attention, "qkv", QKV_BYTES)
        add_edge(edges, attention, proj, "attention_output", TOKEN_BYTES)
        add_edge(edges, proj, residual1, "projected_attention", TOKEN_BYTES)
        add_edge(edges, residual1, norm2, "attention_residual", TOKEN_BYTES)
        add_edge(edges, residual1, residual2, "residual_skip_2", TOKEN_BYTES)
        add_edge(edges, norm2, fc1, "normalized_tokens", TOKEN_BYTES)
        add_edge(edges, fc1, gelu, "ffn_expanded", MLP_BYTES)
        add_edge(edges, gelu, fc2, "activated_ffn", MLP_BYTES)
        add_edge(edges, fc2, residual2, "ffn_output", TOKEN_BYTES)
        previous = residual2

    add_task(tasks, "final_norm", "layer_norm", ["GPU"], [1, TOKENS, HIDDEN], [1, TOKENS, HIDDEN], raw_groups)
    add_task(tasks, "class_token_select", "class_token_select", ["GPU"], [1, TOKENS, HIDDEN], [1, HIDDEN], raw_groups)
    add_task(tasks, "classifier", "linear_classifier", ["GPU", "FPGA"], [1, HIDDEN], [1, CLASSES], raw_groups)
    add_edge(edges, previous, "final_norm", "encoder_output", TOKEN_BYTES)
    add_edge(edges, "final_norm", "class_token_select", "normalized_tokens", TOKEN_BYTES)
    add_edge(edges, "class_token_select", "classifier", "class_token", CLASS_TOKEN_BYTES)

    acyclic = validate_dag(tasks, edges)
    empty_groups = [task["id"] for task in tasks if not task["source_onnx_node_ids"]]
    grouped_raw_nodes = sum(len(task["source_onnx_node_ids"]) for task in tasks)
    expected_groupable = sum(
        node["runtime_task"] and node["op_type"] != "Identity"
        for node in raw_dag["nodes"]
    )
    if grouped_raw_nodes != expected_groupable:
        raise RuntimeError(
            f"Raw-node coverage mismatch: grouped={grouped_raw_nodes}, expected={expected_groupable}"
        )
    if empty_groups:
        raise RuntimeError(f"Coarse tasks without source ONNX nodes: {empty_groups}")

    coarse_dag = {
        "metadata": {
            "source_raw_dag": str(raw_dag_path),
            "model_variant": "ViT-Base/16-like",
            "batch": BATCH,
            "activation_precision": "FP32",
            "weights_assumed_preloaded": True,
            "node_granularity": "coarse ViT scheduling tasks",
        },
        "tasks": tasks,
        "edges": edges,
    }
    dag_path = output_directory / "vit_batch1_coarse_dag.json"
    dag_path.write_text(json.dumps(coarse_dag, indent=2), encoding="utf-8")

    summary = {
        "coarse_tasks": len(tasks),
        "coarse_edges": len(edges),
        "gpu_fpga_linear_tasks": sum(len(task["candidate_devices"]) == 2 for task in tasks),
        "gpu_only_tasks": sum(task["candidate_devices"] == ["GPU"] for task in tasks),
        "grouped_runtime_onnx_nodes": grouped_raw_nodes,
        "ignored_weight_identity_nodes": ignored_identity_nodes,
        "empty_coarse_groups": len(empty_groups),
        "acyclic": acyclic,
        "output": str(dag_path),
    }
    summary_path = output_directory / "coarse_dag_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
