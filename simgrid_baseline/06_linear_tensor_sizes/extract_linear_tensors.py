import csv
import math
import sys
from pathlib import Path

import onnx


def shape_from_value_info(value_info):
    return [
        dimension.dim_value or dimension.dim_param or "?"
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def elements_at_batch_one(shape):
    resolved = []
    for dimension in shape:
        if isinstance(dimension, int):
            resolved.append(dimension)
        elif isinstance(dimension, str):
            # ONNX shape inference may rename the dynamic batch to `unk__N`.
            resolved.append(1)
    return math.prod(resolved)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: python {sys.argv[0]} <model.onnx> <output.csv>")

    model_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(model_path)
    inferred = onnx.shape_inference.infer_shapes(model)
    initializers = {initializer.name: initializer for initializer in inferred.graph.initializer}
    pos_embed = next(
        initializer
        for name, initializer in initializers.items()
        if "pos_embed" in name
    )
    token_count = pos_embed.dims[1]
    value_shapes = {
        value.name: shape_from_value_info(value)
        for value in [
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        ]
    }

    rows = []
    for index, node in enumerate(inferred.graph.node):
        if node.op_type not in {"MatMul", "Gemm"}:
            continue
        weight_name = next((name for name in node.input if name in initializers), None)
        if weight_name is None or len(initializers[weight_name].dims) != 2:
            continue

        weight_shape = list(initializers[weight_name].dims)
        activation_name = next(name for name in node.input if name != weight_name)
        raw_input_shape = value_shapes.get(activation_name, [])
        raw_output_shape = value_shapes.get(node.output[0], [])

        if "/blocks/" in node.name and node.op_type == "MatMul":
            input_shape = [1, token_count, weight_shape[0]]
            output_shape = [1, token_count, weight_shape[1]]
        elif node.op_type == "Gemm":
            input_shape = [1, weight_shape[1]]
            output_shape = [1, weight_shape[0]]
        else:
            input_shape = raw_input_shape
            output_shape = raw_output_shape

        input_elements = elements_at_batch_one(input_shape)
        output_elements = elements_at_batch_one(output_shape)
        if input_elements is None or output_elements is None:
            raise RuntimeError(f"Unresolved shape for {node.name}: {input_shape} -> {output_shape}")

        rows.append(
            {
                "node_index": index,
                "node_name": node.name or f"{node.op_type}_{index}",
                "op_type": node.op_type,
                "raw_onnx_input_shape": "x".join(str(value) for value in raw_input_shape),
                "raw_onnx_output_shape": "x".join(str(value) for value in raw_output_shape),
                "input_shape_batch1": "x".join(str(value) for value in input_shape),
                "output_shape_batch1": "x".join(str(value) for value in output_shape),
                "input_bytes_fp32_batch1": input_elements * 4,
                "output_bytes_fp32_batch1": output_elements * 4,
                "round_trip_bytes_fp32_batch1": (input_elements + output_elements) * 4,
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    unique_shapes = {}
    for row in rows:
        key = (
            row["input_shape_batch1"],
            row["output_shape_batch1"],
            row["round_trip_bytes_fp32_batch1"],
        )
        unique_shapes[key] = unique_shapes.get(key, 0) + 1

    print(f"model={model_path}")
    print(f"linear_candidates={len(rows)}")
    print("shape,count,round_trip_bytes_fp32_batch1")
    for (input_shape, output_shape, round_trip_bytes), count in sorted(unique_shapes.items()):
        print(f"{input_shape}->{output_shape},{count},{round_trip_bytes}")


if __name__ == "__main__":
    main()
