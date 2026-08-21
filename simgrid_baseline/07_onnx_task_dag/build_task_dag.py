import json
import math
import sys
from collections import deque
from pathlib import Path

import onnx
from onnx import TensorProto


ELEMENT_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.UINT8: 1,
    TensorProto.INT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.BFLOAT16: 2,
}


def fix_batch_to_one(model):
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for graph_input in model.graph.input:
        if graph_input.name in initializer_names:
            continue
        dimensions = graph_input.type.tensor_type.shape.dim
        if dimensions:
            dimensions[0].ClearField("dim_param")
            dimensions[0].dim_value = 1


def collect_tensor_metadata(model):
    metadata = {}
    values = [*model.graph.input, *model.graph.value_info, *model.graph.output]
    for value in values:
        tensor_type = value.type.tensor_type
        shape = []
        resolved = True
        for dimension in tensor_type.shape.dim:
            if dimension.HasField("dim_value"):
                shape.append(dimension.dim_value)
            else:
                shape.append(dimension.dim_param or "?")
                resolved = False
        element_bytes = ELEMENT_BYTES.get(tensor_type.elem_type)
        byte_count = math.prod(shape) * element_bytes if resolved and element_bytes else None
        metadata[value.name] = {
            "shape": shape,
            "element_type": TensorProto.DataType.Name(tensor_type.elem_type),
            "bytes_batch1": byte_count,
        }

    for initializer in model.graph.initializer:
        element_bytes = ELEMENT_BYTES.get(initializer.data_type)
        metadata[initializer.name] = {
            "shape": list(initializer.dims),
            "element_type": TensorProto.DataType.Name(initializer.data_type),
            "bytes_batch1": math.prod(initializer.dims) * element_bytes if element_bytes else None,
        }

    pos_embed = next(
        initializer
        for name, initializer in ((item.name, item) for item in model.graph.initializer)
        if "pos_embed" in name
    )
    token_count = pos_embed.dims[1]
    hidden_dim = pos_embed.dims[2]
    vit_shape_overrides = {
        "/Expand_output_0": [1, 1, hidden_dim],
        "/Concat_1_output_0": [1, token_count, hidden_dim],
    }
    for tensor_name, shape in vit_shape_overrides.items():
        if tensor_name in metadata and metadata[tensor_name]["bytes_batch1"] is None:
            metadata[tensor_name] = {
                "shape": shape,
                "element_type": "FLOAT",
                "bytes_batch1": math.prod(shape) * 4,
            }
    return metadata


def is_acyclic(node_count, edges):
    successors = [[] for _ in range(node_count)]
    indegree = [0] * node_count
    for edge in edges:
        source = edge["source_index"]
        target = edge["target_index"]
        successors[source].append(target)
        indegree[target] += 1

    ready = deque(index for index, degree in enumerate(indegree) if degree == 0)
    visited = 0
    while ready:
        source = ready.popleft()
        visited += 1
        for target in successors[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited == node_count


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: python {sys.argv[0]} <model.onnx> <output-directory>")

    model_path = Path(sys.argv[1]).resolve()
    output_directory = Path(sys.argv[2]).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    fix_batch_to_one(model)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    tensor_metadata = collect_tensor_metadata(model)
    producer_by_tensor = {}
    for index, node in enumerate(model.graph.node):
        for output_name in node.output:
            producer_by_tensor[output_name] = index

    edges = []
    predecessors = [set() for _ in model.graph.node]
    successors = [set() for _ in model.graph.node]
    for target_index, node in enumerate(model.graph.node):
        for input_name in node.input:
            source_index = producer_by_tensor.get(input_name)
            if source_index is None:
                continue
            metadata = tensor_metadata.get(input_name, {})
            edges.append(
                {
                    "source": f"n{source_index:04d}",
                    "target": f"n{target_index:04d}",
                    "source_index": source_index,
                    "target_index": target_index,
                    "tensor": input_name,
                    "shape_batch1": metadata.get("shape"),
                    "bytes_batch1": metadata.get("bytes_batch1"),
                }
            )
            predecessors[target_index].add(source_index)
            successors[source_index].add(target_index)

    nodes = []
    linear_count = 0
    for index, node in enumerate(model.graph.node):
        weight = next((initializers[name] for name in node.input if name in initializers), None)
        is_linear = (
            node.op_type in {"MatMul", "Gemm"}
            and weight is not None
            and len(weight.dims) == 2
        )
        linear_count += int(is_linear)
        runtime_task = node.op_type != "Constant"
        activation_inputs = [name for name in node.input if name not in initializers]
        nodes.append(
            {
                "id": f"n{index:04d}",
                "index": index,
                "name": node.name or f"{node.op_type}_{index}",
                "op_type": node.op_type,
                "runtime_task": runtime_task,
                "offloadable_linear": is_linear,
                "candidate_devices": ["GPU", "FPGA"] if is_linear else (["GPU"] if runtime_task else []),
                "activation_inputs": activation_inputs,
                "outputs": list(node.output),
                "predecessors": [f"n{value:04d}" for value in sorted(predecessors[index])],
                "successors": [f"n{value:04d}" for value in sorted(successors[index])],
            }
        )

    dag = {
        "metadata": {
            "source_model": str(model_path),
            "batch": 1,
            "node_granularity": "raw ONNX operators",
            "weights_assumed_preloaded": True,
            "target_devices": ["GPU", "FPGA"],
        },
        "nodes": nodes,
        "edges": [
            {key: value for key, value in edge.items() if key not in {"source_index", "target_index"}}
            for edge in edges
        ],
    }
    output_path = output_directory / "vit_batch1_raw_dag.json"
    output_path.write_text(json.dumps(dag, indent=2), encoding="utf-8")

    known_edges = sum(edge["bytes_batch1"] is not None for edge in edges)
    summary = {
        "onnx_nodes": len(nodes),
        "runtime_nodes": sum(node["runtime_task"] for node in nodes),
        "constant_nodes": sum(not node["runtime_task"] for node in nodes),
        "offloadable_linear_nodes": linear_count,
        "tensor_dependency_edges": len(edges),
        "edges_with_known_batch1_bytes": known_edges,
        "edges_with_unresolved_bytes": len(edges) - known_edges,
        "acyclic": is_acyclic(len(nodes), edges),
        "output": str(output_path),
    }
    summary_path = output_directory / "dag_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
