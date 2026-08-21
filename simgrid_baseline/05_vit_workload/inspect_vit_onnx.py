import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

import onnx


def tensor_shape(value_info):
    dimensions = value_info.type.tensor_type.shape.dim
    return [dimension.dim_value or dimension.dim_param or "?" for dimension in dimensions]


def classify_vit(hidden_dim, block_count, parameter_count):
    if hidden_dim == 384 and block_count == 12:
        return "ViT-Small/16-like"
    if hidden_dim == 768 and block_count == 12:
        return "ViT-Base/16-like"
    return "Unknown ViT variant"


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: python {sys.argv[0]} <model.onnx> <output-directory>")

    model_path = Path(sys.argv[1]).resolve()
    output_directory = Path(sys.argv[2]).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    operation_counts = Counter(node.op_type for node in model.graph.node)
    parameter_count = sum(math.prod(initializer.dims) for initializer in initializers.values())

    pos_embed = next(
        (initializer for name, initializer in initializers.items() if "pos_embed" in name),
        None,
    )
    hidden_dim = pos_embed.dims[-1] if pos_embed and pos_embed.dims else None
    block_indices = {
        int(match.group(1))
        for node in model.graph.node
        if (match := re.search(r"/blocks/blocks\.(\d+)/", node.name))
    }
    block_count = max(block_indices) + 1 if block_indices else 0

    linear_rows = []
    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"MatMul", "Gemm"}:
            continue
        weight = next((initializers[name] for name in node.input if name in initializers), None)
        if weight is None or len(weight.dims) != 2:
            continue
        weight_shape = list(weight.dims)
        linear_rows.append(
            {
                "node_index": index,
                "node_name": node.name or f"{node.op_type}_{index}",
                "op_type": node.op_type,
                "weight_name": weight.name,
                "weight_shape": "x".join(str(value) for value in weight_shape),
                "weight_parameters": math.prod(weight_shape),
                "weight_bytes_fp32": math.prod(weight_shape) * 4,
            }
        )

    with (output_directory / "linear_layers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=linear_rows[0].keys())
        writer.writeheader()
        writer.writerows(linear_rows)

    model_variant = classify_vit(hidden_dim, block_count, parameter_count)
    summary_lines = [
        f"model_path={model_path}",
        f"file_size_bytes={model_path.stat().st_size}",
        f"graph_inputs={[(item.name, tensor_shape(item)) for item in model.graph.input]}",
        f"graph_outputs={[(item.name, tensor_shape(item)) for item in model.graph.output]}",
        f"parameter_count={parameter_count}",
        f"hidden_dim={hidden_dim}",
        f"transformer_blocks={block_count}",
        f"linear_nodes_with_constant_weights={len(linear_rows)}",
        f"inferred_variant={model_variant}",
        "operation_counts=" + ",".join(f"{name}:{count}" for name, count in sorted(operation_counts.items())),
    ]
    summary = "\n".join(summary_lines) + "\n"
    (output_directory / "model_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
