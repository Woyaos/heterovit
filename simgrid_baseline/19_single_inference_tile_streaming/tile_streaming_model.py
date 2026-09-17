#!/usr/bin/env python3
"""Single-request tiled FFN dataflow sensitivity model.

This model does not predict RF880 performance. It turns the existing whole-FFN
sensitivity costs into a finite-buffer, resource-constrained tile schedule so
that architectures can be rejected before RTL work starts.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "simgrid_baseline/17_rf880_agx_final/rf880_agx_sensitivity_profile.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def transfer_stages(profile):
    path = profile["communication_path"]
    if path["kind"] == "host_staged_two_copy":
        return [
            ("gpu_host", path["gpu_host"]),
            ("host_fpga_pcie", path["host_fpga_pcie"]),
        ]
    if path["kind"] in {"shared_mapped_one_copy", "direct_dma_one_copy"}:
        return [("host_fpga_pcie", path["host_fpga_pcie"])]
    raise ValueError(f"Unsupported communication path: {path['kind']}")


def stage_transfer_us(byte_count, stage, include_fixed=True):
    fixed = stage["fixed_latency_us"] if include_fixed else 0.0
    return fixed + byte_count / (stage["effective_bandwidth_GBps"] * 1000.0)


def ffn_parameters(profile, tokens=197, hidden=768, expansion=4, weight_bits=2):
    bpe = profile["activation"]["bytes_per_element"]
    expanded = hidden * expansion
    gpu = profile["gpu"]
    fpga = profile["fpga"]
    linear_flops = 2 * tokens * hidden * expanded
    one_linear_gpu_compute_us = linear_flops / (gpu["matmul_effective_tflops"] * 1_000_000.0)
    one_linear_fpga_compute_us = one_linear_gpu_compute_us / fpga["linear_speedup_over_gpu"]
    expanded_bytes = tokens * expanded * bpe
    gelu_gpu_compute_us = (2 * expanded_bytes) / (gpu["memory_bandwidth_GBps"] * 1000.0)
    gelu_fpga_compute_us = gelu_gpu_compute_us / fpga.get("elementwise_speedup_over_gpu", 1.0)
    return {
        "tokens": tokens,
        "hidden": hidden,
        "expanded": expanded,
        "expansion": expansion,
        "bytes_per_element": bpe,
        "weight_bits": weight_bits,
        "input_bytes": tokens * hidden * bpe,
        "expanded_bytes": expanded_bytes,
        "weight_bytes": math.ceil(2 * hidden * expanded * weight_bits / 8),
        "dispatch_us": fpga["dispatch_overhead_us"],
        "fc1_total_us": one_linear_fpga_compute_us,
        "gelu_total_us": gelu_fpga_compute_us,
        "fc2_total_us": one_linear_fpga_compute_us,
    }


def tile_counts(tokens, tile_tokens):
    return [min(tile_tokens, tokens - start) for start in range(0, tokens, tile_tokens)]


def conservative_buffer_bytes(params, tile_tokens, buffers):
    tile = min(tile_tokens, params["tokens"])
    # Input, two expanded intermediates, and output, each with B banks.
    activations = buffers * tile * params["bytes_per_element"] * (
        2 * params["hidden"] + 2 * params["expanded"]
    )
    return {
        "activation_buffers": activations,
        "resident_low_bit_weights": params["weight_bytes"],
        "total": activations + params["weight_bytes"],
    }


def whole_tensor_latency_us(profile, params):
    io = sum(stage_transfer_us(params["input_bytes"], stage) for _, stage in transfer_stages(profile))
    compute = (
        params["dispatch_us"]
        + params["fc1_total_us"]
        + params["gelu_total_us"]
        + params["fc2_total_us"]
    )
    return 2 * io + compute


def internal_dataflow_latency_us(profile, params, tile_tokens, independent_engines=True):
    """Whole-tensor DMA followed by an on-chip token-tile dataflow pipeline."""
    io = sum(stage_transfer_us(params["input_bytes"], stage) for _, stage in transfer_stages(profile))
    available = {"fc1": 0.0, "gelu": 0.0, "fc2": 0.0, "shared": 0.0}
    final = 0.0
    for count in tile_counts(params["tokens"], tile_tokens):
        scale = count / params["tokens"]
        predecessor = 0.0
        for name in ("fc1", "gelu", "fc2"):
            resource = name if independent_engines else "shared"
            duration = params[f"{name}_total_us"] * scale
            start = max(predecessor, available[resource])
            predecessor = start + duration
            available[resource] = predecessor
        final = predecessor
    return io + params["dispatch_us"] + final + io


def _resource_name(base, direction, links_share_both_directions):
    return base if links_share_both_directions else f"{base}:{direction}"


def end_to_end_stream_latency_us(
    profile,
    params,
    tile_tokens,
    buffers=2,
    independent_engines=True,
    dma_mode="streaming_descriptor",
):
    """List-schedule one FFN with finite input/output tile buffers.

    streaming_descriptor pays each transfer-stage startup once per tensor and
    direction. per_tile_dma pays it for every tile, exposing the small-transfer
    penalty that a real descriptor-per-tile implementation would incur.
    """
    if buffers < 1:
        raise ValueError("buffers must be at least one")
    if dma_mode not in {"streaming_descriptor", "per_tile_dma"}:
        raise ValueError(f"Unsupported dma_mode: {dma_mode}")

    stages = transfer_stages(profile)
    counts = tile_counts(params["tokens"], tile_tokens)
    operations = {}
    last_input = []
    first_output = []
    last_output = []

    for tile_id, count in enumerate(counts):
        tile_bytes = count * params["hidden"] * params["bytes_per_element"]
        scale = count / params["tokens"]
        previous = None
        input_ops = []
        for stage_id, (base, stage) in enumerate(stages):
            op_id = f"t{tile_id}:in:{stage_id}"
            include_fixed = dma_mode == "per_tile_dma" or tile_id == 0
            deps = [] if previous is None else [previous]
            operations[op_id] = {
                "deps": deps,
                "resource": _resource_name(
                    base, "in", profile["communication_path"].get("links_share_both_directions", True)
                ),
                "duration": stage_transfer_us(tile_bytes, stage, include_fixed),
                "priority": 0,
            }
            previous = op_id
            input_ops.append(op_id)
        last_input.append(previous)

        for name, priority in (("fc1", 1), ("gelu", 2), ("fc2", 3)):
            op_id = f"t{tile_id}:{name}"
            resource = name if independent_engines else "shared_compute"
            operations[op_id] = {
                "deps": [previous],
                "resource": resource,
                "duration": params[f"{name}_total_us"] * scale,
                "priority": priority,
            }
            previous = op_id

        first_output.append(previous)
        output_ops = []
        for reverse_id, (base, stage) in enumerate(reversed(stages)):
            op_id = f"t{tile_id}:out:{reverse_id}"
            include_fixed = dma_mode == "per_tile_dma" or tile_id == 0
            operations[op_id] = {
                "deps": [previous],
                "resource": _resource_name(
                    base, "out", profile["communication_path"].get("links_share_both_directions", True)
                ),
                "duration": stage_transfer_us(tile_bytes, stage, include_fixed),
                "priority": 4,
            }
            previous = op_id
            output_ops.append(op_id)
        last_output.append(previous)

    # Banks are finite at every tile boundary. Release is conservatively placed
    # at consumer completion because this operation-level model does not expose
    # word-level FIFO reads.
    for tile_id in range(buffers, len(counts)):
        operations[f"t{tile_id}:in:0"]["deps"].append(f"t{tile_id - buffers}:fc1")
        operations[f"t{tile_id}:fc1"]["deps"].append(f"t{tile_id - buffers}:gelu")
        operations[f"t{tile_id}:gelu"]["deps"].append(f"t{tile_id - buffers}:fc2")
        operations[f"t{tile_id}:fc2"]["deps"].append(last_output[tile_id - buffers])

    dependents = {op_id: [] for op_id in operations}
    indegree = {op_id: len(op["deps"]) for op_id, op in operations.items()}
    for op_id, op in operations.items():
        for dep in op["deps"]:
            dependents[dep].append(op_id)

    ready = []
    for op_id, degree in indegree.items():
        if degree == 0:
            heapq.heappush(ready, (0.0, operations[op_id]["priority"], op_id))

    finish = {}
    resource_available = {}
    timeline = []
    while ready:
        _hint, _priority, op_id = heapq.heappop(ready)
        op = operations[op_id]
        dependency_finish = max((finish[dep] for dep in op["deps"]), default=0.0)
        start = max(dependency_finish, resource_available.get(op["resource"], 0.0))
        end = start + op["duration"]
        finish[op_id] = end
        resource_available[op["resource"]] = end
        timeline.append({"operation": op_id, "resource": op["resource"], "start_us": start, "finish_us": end})
        for child in dependents[op_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                child_ready = max(finish[dep] for dep in operations[child]["deps"])
                heapq.heappush(ready, (child_ready, operations[child]["priority"], child))

    if len(finish) != len(operations):
        raise RuntimeError("Tile schedule contains a dependency cycle")
    makespan = params["dispatch_us"] + max(finish.values(), default=0.0)
    return makespan, timeline


def run_sweep(profile, tile_sizes, buffers_list, weight_bits):
    params = ffn_parameters(profile, weight_bits=weight_bits)
    monolithic = whole_tensor_latency_us(profile, params)
    rows = []
    for tile in tile_sizes:
        for buffers in buffers_list:
            memory = conservative_buffer_bytes(params, tile, buffers)
            for independent in (False, True):
                architecture = "independent_fc_engines" if independent else "shared_fc_engine"
                internal = internal_dataflow_latency_us(profile, params, tile, independent)
                rows.append({
                    "mode": "internal_dataflow_whole_tensor_dma",
                    "dma_mode": "whole_tensor",
                    "architecture": architecture,
                    "tile_tokens": tile,
                    "buffers": buffers,
                    "ffn_latency_us": internal,
                    "speedup_vs_monolithic": monolithic / internal,
                    "conservative_buffer_bytes": memory["activation_buffers"],
                    "resident_weight_bytes": memory["resident_low_bit_weights"],
                    "total_local_bytes": memory["total"],
                })
                for dma_mode in ("per_tile_dma", "streaming_descriptor"):
                    latency, _ = end_to_end_stream_latency_us(
                        profile, params, tile, buffers, independent, dma_mode
                    )
                    rows.append({
                        "mode": "end_to_end_tile_stream",
                        "dma_mode": dma_mode,
                        "architecture": architecture,
                        "tile_tokens": tile,
                        "buffers": buffers,
                        "ffn_latency_us": latency,
                        "speedup_vs_monolithic": monolithic / latency,
                        "conservative_buffer_bytes": memory["activation_buffers"],
                        "resident_weight_bytes": memory["resident_low_bit_weights"],
                        "total_local_bytes": memory["total"],
                    })
    return params, monolithic, rows


def write_outputs(output_dir, profile, params, monolithic, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output_dir / "tile_sweep.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    optimistic_best = min(rows, key=lambda row: row["ffn_latency_us"])
    bounded_candidates = [
        row
        for row in rows
        if row["architecture"] == "independent_fc_engines"
        and row["dma_mode"] == "streaming_descriptor"
        and row["buffers"] == 2
        and row["tile_tokens"] in {8, 16, 32, 64}
    ]
    best_bounded = min(
        (
            row
            for row in bounded_candidates
        ),
        key=lambda row: row["ffn_latency_us"],
    )
    initial_hls_candidate = next(row for row in bounded_candidates if row["tile_tokens"] == 16)
    existing_full_model_us = 17850.05792000002
    summary = {
        "status": "unmeasured_sensitivity_analysis",
        "valid_for_target_prediction": False,
        "profile_name": profile["profile_name"],
        "model": params,
        "whole_tensor_ffn_latency_us": monolithic,
        "optimistic_lower_bound_case": optimistic_best,
        "best_pre_rtl_sweep_case": best_bounded,
        "recommended_initial_hls_candidate": initial_hls_candidate,
        "pre_rtl_candidates": bounded_candidates,
        "conditional_full_model_us": existing_full_model_us
        - 12 * (monolithic - initial_hls_candidate["ffn_latency_us"]),
        "hardware_requirements": [
            "FC1, GELU, and FC2 must be concurrent FPGA processes for independent_fc_engines.",
            "Weights must remain resident or their load cost must be added.",
            "streaming_descriptor requires one transfer setup with chunk-level production/consumption.",
            "DMA, FPGA compute, and reverse transfer must be able to overlap as modeled.",
            "The conservative local-memory total must fit after routing and other RF880 logic reservations.",
        ],
        "limitations": [
            "All compute and communication rates are sensitivity assumptions, not board measurements.",
            "The model optimizes one FFN inside one inference; attention remains a cross-layer barrier.",
            "Tile-control and FIFO handshake overheads are zero until HLS reports are available.",
            "It does not model RTL initiation intervals, timing closure, DDR arbitration, or numerical accuracy.",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")

    report = f"""# Single-Inference Tile Streaming Sensitivity Result

- Status: unmeasured sensitivity analysis
- Valid for RF880 target prediction: false
- Whole-FFN baseline: {monolithic:.3f} us
- Recommended first HLS point: {initial_hls_candidate['ffn_latency_us']:.3f} us
- Conditional FFN speedup at that point: {initial_hls_candidate['speedup_vs_monolithic']:.3f}x
- Tile size: {initial_hls_candidate['tile_tokens']} tokens
- Buffers: {initial_hls_candidate['buffers']}
- DMA model: {initial_hls_candidate['dma_mode']}
- Compute architecture: {initial_hls_candidate['architecture']}
- Conservative activation buffers: {initial_hls_candidate['conservative_buffer_bytes'] / 1024:.1f} KiB
- Resident low-bit weights ({params['weight_bits']} bit): {initial_hls_candidate['resident_weight_bytes'] / 1024:.1f} KiB
- Conditional full-model latency: {summary['conditional_full_model_us'] / 1000:.3f} ms

The 16-token point is an initial synthesis candidate, not a proven optimum. The
conditional full-model value replaces each of the twelve existing monolithic
FPGA FFN intervals with that tile-stream result. It is not an RF880 prediction.
Per-tile DMA rows deliberately pay fixed startup for every tile; they show when
fine tiles lose to whole-tensor transfers. Hardware acceptance requires HLS/RTL
reports and measured DMA overlap.
"""
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--tile-sizes", default="1,2,4,8,16,32,64,197")
    parser.add_argument("--buffers", default="1,2")
    parser.add_argument("--weight-bits", type=int, default=2)
    args = parser.parse_args()
    profile = load_json(args.profile)
    tile_sizes = [int(value) for value in args.tile_sizes.split(",")]
    buffers = [int(value) for value in args.buffers.split(",")]
    params, monolithic, rows = run_sweep(profile, tile_sizes, buffers, args.weight_bits)
    summary = write_outputs(args.output, profile, params, monolithic, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
